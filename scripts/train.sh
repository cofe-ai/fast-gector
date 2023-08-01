#!/bin/bash
WANDB_KEY=$WANDB_KEY # if wandb is enabled, set WANDB_KEY to log in
detect_vocab_path="./data/vocabulary/d_tags.txt"
correct_vocab_path="./data/vocabulary/labels.txt"
train_path="train.edits"
valid_path="dev.edits"
config_path="configs/ds_config_basic.json"
timestamp=`date "+%Y%0m%0d_%T"`
save_dir="../ckpts/ckpt_$timestamp"
tensorboard_dir="log/tb/gector_${timestamp}"
pretrained_transformer_path="./pretrained_models/bert-base-uncased"
mkdir -p $save_dir
cp $0 $save_dir
cp $config_path $save_dir


run_cmd="deepspeed --hostfile configs/hostfile --master_port 49828 train.py \
    --deepspeed \
    --deepspeed_config $config_path \
    --num_epochs 1 \
    --max_len 256 \
    --valid_batch_size 256 \
    --cold_step_count 0 \
    --warmup 0.1 \
    --cold_lr 1e-3 \
    --skip_correct 0 \
    --skip_complex 0 \
    --sub_token_mode average \
    --special_tokens_fix 1 \
    --unk2keep 0 \
    --tp_prob 1 \
    --tn_prob 1 \
    --detect_vocab_path $detect_vocab_path \
    --correct_vocab_path $correct_vocab_path \
    --do_eval \
    --train_path $train_path \
    --valid_path $valid_path \
    --use_cache 1 \
    --save_dir $save_dir \
    --pretrained_transformer_path $pretrained_transformer_path \
    --tensorboard_dir $tensorboard_dir \
    2>&1 | tee ${save_dir}/train-${timestamp}.log"

echo ${run_cmd}
eval ${run_cmd}
