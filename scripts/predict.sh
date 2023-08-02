#!/bin/bash
pretrained_transformer_path="roberta-base"
ckpt_path="ckpts/globalstep-xxxx"
input_path="test.src"
out_path="test.pred"
mkdir result
deepspeed --include localhost:0 --master_port 42991 predict.py \
    --batch_size 256 \
    --iteration_count 5 \
    --min_seq_len 3 \
    --max_num_tokens 128 \
    --min_error_probability 0.0 \
    --additional_confidence 0.0 \
    --sub_token_mode "average" \
    --max_pieces_per_token 5 \
    --ckpt_path $ckpt_path \
    --deepspeed_config "./configs/ds_config_zero1_fp16.json" \
    --detect_vocab_path "./data/vocabulary/d_tags.txt" \
    --correct_vocab_path "./data/vocabulary/labels.txt" \
    --pretrained_transformer_path $pretrained_transformer_path \
    --input_path $input_path \
    --out_path $out_path \
    --special_tokens_fix 1 \
    --detokenize 0 \
    --segmented 1 \
    2>&1 | tee debug.log