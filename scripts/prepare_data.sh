#!/bin/bash
SUBSET="train-stage2"
SOURCE="../gec_private_train_data/${SUBSET}.src"
TARGET="../gec_private_train_data/${SUBSET}.trg"
OUTPUT="../gec_private_train_data/${SUBSET}.edits"
python utils/preprocess_data.py -s $SOURCE -t $TARGET -o $OUTPUT