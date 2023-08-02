# -*- coding:UTF-8 -*-
import re
from transformers import AutoTokenizer
from utils.mismatched_utils import *
from src.dataset import Seq2EditVocab, MyCollate
from utils.helpers import INCORRECT_LABEL, KEEP_LABEL, PAD_LABEL, START_TOKEN, UNK_LABEL, get_target_sent_by_edits
from src.model import GECToRModel
from random import seed
import deepspeed
import os

class Predictor:
    def __init__(self, args):
        self.fix_seed()
        deepspeed.init_distributed()
        self.device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.iteration_count = args.iteration_count
        self.min_seq_len = args.min_seq_len
        self.max_num_tokens = args.max_num_tokens
        self.min_error_probability = args.min_error_probability
        self.max_pieces_per_token = args.max_pieces_per_token
        self.vocab = Seq2EditVocab(
            args.detect_vocab_path, args.correct_vocab_path, unk2keep=bool(args.unk2keep))
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_transformer_path, do_basic_tokenize=False)
        self.base_tokenizer_vocab = self.base_tokenizer.get_vocab()
        if bool(args.special_tokens_fix):  # for roberta
            self.base_tokenizer.add_tokens([START_TOKEN], special_tokens=True)
            # set start_token to unk_token_id is no longer supported via transformers tokenizer
            # since access the vocab is implemented by calling get_vocab() which create a new instance,
            # in this case, we cannot actually change the vocab.
            # Instead, we can get the vocab and change it, then use it directly later on.
            # self.base_tokenizer.vocab[START_TOKEN] = self.base_tokenizer.unk_token_id
            self.base_tokenizer_vocab[START_TOKEN] = self.base_tokenizer.unk_token_id
        self.mismatched_tokenizer = MisMatchedTokenizer(
            self.base_tokenizer, self.base_tokenizer_vocab, self.max_pieces_per_token)
        self.collate_fn = MyCollate(
            max_len=self.max_num_tokens,
            input_pad_id=self.base_tokenizer.pad_token_id,
            detect_pad_id=self.vocab.detect_vocab["tag2id"][PAD_LABEL],
            correct_pad_id=self.vocab.correct_vocab["tag2id"][PAD_LABEL])
        self.model = self.init_model(args)
        self.model.eval()

    def init_model(self, args):
        model = GECToRModel(
            encoder_path=args.pretrained_transformer_path,
            num_detect_tags=len(self.vocab.detect_vocab["id2tag"]),
            num_correct_tags=len(self.vocab.correct_vocab["id2tag"]),
            additional_confidence=args.additional_confidence,
            dp_rate=0.0,
            detect_pad_id=self.vocab.detect_vocab["tag2id"][PAD_LABEL],
            correct_pad_id=self.vocab.correct_vocab["tag2id"][PAD_LABEL],
            detect_incorrect_id=self.vocab.detect_vocab["tag2id"][INCORRECT_LABEL],
            correct_keep_id=self.vocab.correct_vocab["tag2id"][KEEP_LABEL],
            sub_token_mode=args.sub_token_mode,
            device=self.device
        )
        ds_engine, _, _, _ = deepspeed.initialize(
            args=args, model=model, model_parameters=model.parameters())
        load_dir, tag = os.path.split(args.ckpt_path)
        ds_engine.load_checkpoint(load_dir=load_dir, tag=tag, load_module_only=True, load_optimizer_states=False, load_lr_scheduler_states=False)

        return ds_engine

    def handle_batch(self, full_batch):
        final_batch = full_batch[:]
        # {sent idx: sent}, used for stop iter early
        prev_preds_dict = {idx: [sent] for idx, sent in enumerate(final_batch)}
        short_skip_id_set = set([idx for idx, sent in enumerate(
            final_batch) if len(sent) < self.min_seq_len])
        # idxs for len(sent) > min_seq_len
        pred_ids = [idx for idx in range(
            len(full_batch)) if idx not in short_skip_id_set]
        total_updates = 0

        for n_iter in range(self.iteration_count):
            ori_batch = [final_batch[i] for i in pred_ids]
            batch_input_dict, truncated_seq_lengths = self.preprocess(ori_batch)
            if not batch_input_dict:
                break
            label_probs, label_ids, max_detect_incor_probs = self.predict(
                batch_input_dict)
            del batch_input_dict
            # list of sents(each sent is a list of target tokens)
            pred_batch = self.postprocess(
                ori_batch, truncated_seq_lengths, label_probs, label_ids, max_detect_incor_probs)

            final_batch, pred_ids, cnt = \
                self.update_final_batch(final_batch, pred_ids, pred_batch,
                                        prev_preds_dict)
            total_updates += cnt
            if not pred_ids:
                break
        return final_batch, total_updates

    def predict(self, batch_inputs):
        with torch.no_grad():
            for k, v in batch_inputs.items():
                batch_inputs[k] = v.cuda()
            outputs = self.model(batch_inputs)
        label_probs, label_ids = torch.max(
            outputs['class_probabilities_labels'], dim=-1)
        max_detect_incor_probs = outputs['max_error_probability']
        return label_probs.tolist(), label_ids.tolist(), max_detect_incor_probs.tolist()

    def preprocess(self, seqs):
        seq_lens = [len(seq) for seq in seqs if seq]
        if not seq_lens:
            return []
        input_dict_batch = []
        truncated_seq_lengths = []
        for words in seqs:
            words = [START_TOKEN] + words
            input_ids, offsets, truncated_seq_length = self.mismatched_tokenizer.encode(words, add_special_tokens=False, max_tokens=self.max_num_tokens)
            words = words[:truncated_seq_length]
            truncated_seq_lengths.append(truncated_seq_length)
            input_dict = self.build_input_dict(input_ids, offsets, len(words))
            input_dict_batch.append(input_dict)
        batch_input_dict = self.collate_fn(input_dict_batch)
        for k, v in batch_input_dict.items():
            batch_input_dict[k] = v.to(self.device)
        return batch_input_dict, truncated_seq_lengths

    def postprocess(self, batch, truncated_seq_lengths, batch_label_probs, batch_label_ids, batch_incor_probs):
        keep_id = self.vocab.correct_vocab["tag2id"][KEEP_LABEL]
        all_results = []
        for tokens, truncated_seq_length, label_probs, label_ids, incor_prob in zip(batch, truncated_seq_lengths, batch_label_probs,
                                                              batch_label_ids, batch_incor_probs):
            # since we add special tokens before truncation, max_len should minus 1. This is different from original gector.
            edits = []

            # skip the whole sent if all labels are $KEEP
            if max(label_ids) == keep_id:
                all_results.append(tokens)
                continue

            # if max detect_incor_probs < min_error_prob, skip
            if incor_prob < self.min_error_probability:
                all_results.append(tokens)
                continue

            for idx in range(truncated_seq_length):
                if idx == 0:
                    token = START_TOKEN
                else:
                    # tokens in ori_batch don't have "$START" token, thus offset = 1
                    token = tokens[idx-1]
                if label_ids[idx] == keep_id:
                    continue
                # prediction for \s matched token is $keep, for spellcheck task.
                if re.search("\s+", token):
                    continue
                label = self.vocab.correct_vocab["id2tag"][label_ids[idx]]
                action = self.get_label_action(
                    token, idx, label_probs[idx], label)

                if not action:
                    continue
                edits.append(action)
            # append the target sent (list of target tokens)
            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results

    def update_final_batch(self, final_batch, pred_ids, pred_batch,
                           prev_preds_dict):

        new_pred_ids = []
        total_updated = 0

        for i, ori_id in enumerate(pred_ids):
            ori_tokens = final_batch[ori_id]
            pred_tokens = pred_batch[i]
            prev_preds = prev_preds_dict[ori_id]

            if ori_tokens != pred_tokens:
                if pred_tokens not in prev_preds:
                    final_batch[ori_id] = pred_tokens
                    new_pred_ids.append(ori_id)
                    prev_preds_dict[ori_id].append(pred_tokens)
                else:
                    final_batch[ori_id] = pred_tokens
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def get_label_action(self, token: str, idx: int, label_prob: float, label: str):
        if label_prob < self.min_error_probability or label in [UNK_LABEL, PAD_LABEL, KEEP_LABEL]:
            return None

        if label.startswith("$REPLACE_") or label.startswith("$TRANSFORM_") or label == "$DELETE":
            start_pos = idx
            end_pos = idx + 1
        elif label.startswith("$APPEND_") or label.startswith("$MERGE_"):
            start_pos = idx + 1
            end_pos = idx + 1

        if label == "$DELETE":
            processed_label = ""

        elif label.startswith("$TRANSFORM_") or label.startswith("$MERGE_"):
            processed_label = label[:]

        else:
            processed_label = label[label.index("_")+1:]
        return start_pos - 1, end_pos - 1, processed_label, label_prob

    def build_input_dict(self, input_ids, offsets, word_level_len):
        token_type_ids = [0 for _ in range(len(input_ids))]
        attn_mask = [1 for _ in range(len(input_ids))]
        word_mask = [1 for _ in range(word_level_len)]
        input_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attn_mask,
            "offsets": offsets,
            "word_mask": word_mask}
        return input_dict

    def fix_seed(self):
        torch.manual_seed(1)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        seed(43)
