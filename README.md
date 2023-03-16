# FastGECToR

## Introduction
A faster and simpler implementation of [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://github.com/grammarly/gector) with amp and distributed support by deepspeed. 

Note: To make it faster and more readable, we remove allennlp dependencies and reconstruct related codes.

## Requirements

1. Install Pytorch with cuda support
    ```
    conda create -n gector_env python=3.9 -y
    conda activate gector_env
    conda install pytorch=1.12.1 cudatoolkit==11.3 -c pytorch
    ```

2. Install **NVIDIA-Apex** (for using amp with deepspeed)
    ```bash
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```
3. Install following packages by conda/pip
    ```bash
    transformers==4.26.1
    scikit-learn==1.0.2
    numpy==1.24.2
    deepspeed==0.8.2
    ```

## Preprocess Data
1. Tokenize your data (one sentence per line, split words by space)

2. Generate edits from parallel sents
    ```bash
    bash scripts/prepare_data.sh
    ```

3. \*(Optional) Define your own target vocab (data/vocabulary/labels.txt)

## Train Model
- Edit deepspeed_config.json according to your config params. Note that lr and batch_size options will be overrided by args. And args.lr indicates batch_size (regardless how many gpus are used, which equals effective_batch_size_per_gpu * num_gpus) * num accumulation steps. See more details at src/trainer.py.
   ```bash
   bash scripts/train.sh
   ```

## Inference
- Edit deepspeed_config.json according to your config params
    ```bash
    bash scripts/predict.sh
    ```

## Known Issues
- In distributed training (num gpu > 1), enable AMP with O1 state may raise ZeroDivision Error, which may be caused by apex, see APEX's github issues for help. Or, you can try a smaller lr to see if the error disappears.

## Reference
[1] Omelianchuk, K., Atrasevych, V., Chernodub, A., & Skurzhanskyi, O. (2020). GECToR -- Grammatical Error Correction: Tag, Not Rewrite. arXiv:2005.12592 [cs]. http://arxiv.org/abs/2005.12592

