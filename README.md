# FastGECToR

## Introduction
A faster and simpler implementation of [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://aclanthology.org/2020.bea-1.16/) with amp and distributed support by deepspeed. 

Note: To make it faster and more readable, we remove allennlp dependencies and reconstruct related codes.

## Requirements

1. Install Pytorch with cuda support
    ```
    conda create -n gector_env python=3.7.6 -y
    conda activate gector_env
    conda install pytorch=1.10.1 cudatoolkit -c pytorch
    ```

2. Install **NVIDIA-Apex** (for using amp with deepspeed)
    ```bash
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```
3. Install following packages by conda/pip
    ```bash
    python==3.7.6
    transformers==4.14.1
    scikit-learn==1.0.2
    numpy==1.21.2
    deepspeed==0.5.10
    ```

## Preprocess Data
1. Tokenize your data (one sentence per line, split words by space)

2. Generate edits from parallel sents
    ```bash
    python utils/preprocess_data.py -s source_file -t target_file -o output_edit_file
    ```

3. \*(Optional) Define your own target vocab (data/vocabulary/labels.txt)

## Train Model
- Edit deepspeed_config.json according to your config params. Note that lr and batch_size options will be overrided by args
   ```bash
   bash train.sh
   ```

## Inference
- Edit deepspeed_config.json according to your config params
    ```bash
    bash predict.sh
    ```

## Reference
[1] Omelianchuk, K., Atrasevych, V., Chernodub, A., & Skurzhanskyi, O. (2020). GECToR -- Grammatical Error Correction: Tag, Not Rewrite. arXiv:2005.12592 [cs]. http://arxiv.org/abs/2005.12592

