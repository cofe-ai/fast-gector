# FastGECToR

## 1. Introduction

A faster and simpler implementation of [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://github.com/grammarly/gector) with amp and distributed support by deepspeed.

Note: To make it faster and more readable, we remove allennlp dependencies and reconstruct related codes.

## 2. Requirements

1. Install Pytorch with cuda support `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
2. Install [NVIDIA-Apex](https://github.com/NVIDIA/apex) with CUDA and C++ extensions
3. Install the rest packages with `pip install -r ./requirements.txt`

## 3. Data Processing

1. Tokenize your data (one sentence per line, split words by space)
2. Generate edits from parallel sents `bash scripts/prepare_data.sh`
3. (Optional) Define your own target vocab (data/vocabulary/labels.txt)

## 4. Training

- Edit deepspeed_config.json according to your config params. Edit deepspeed_config.json according to your config params. lr, train_batch_size, gradient_accumulation_steps will be inherited from deepspeed config file.

```bash
bash scripts/train.sh
```

### * Performance Tuning

- Suppose you want to train a GECToR model with bert-base-uncased (with 110M params), and the max seq len is set to 256 for all cases. There’re some configurations you may need to consider in order to achieve better performance / efficiency.
- The basic config is to use single GPU without any tricks. Then you may get the following statistics.
    
    
    | global batch size | n_gpus | MaxMemAllocated (CUDA) | GPU Mem Usage (NVIDIA-SMI) |
    | --- | --- | --- | --- |
    | 8 | 1 | 3.3GB | 5880MiB |
    | 16 | 1 | 5.33GB | 7610MiB |
    | 32 | 1 | 9.28GB | 11712MiB |
    | 64 | 1 | 17.28GB | 20344MiB |
    | 128 | 1 | 33.25GB | 36654MiB |
    | 256 | 1 | 65.21GB | 69864MiB |
- As you can see, The max batch size you can set is limited by the GPU memory allocation. The simplest way to get a larger batch size is to use gradient accumulation, which accumulates the gradients several steps and update at a given interval. In this case, you can reduce the memory usage a lot.
    
    
    | global batch size | effective batch size | gradient accumulation steps | n_gpus | MaxMemAllocated (CUDA) | GPU Mem Usage (NVIDIA-SMI) |
    | --- | --- | --- | --- | --- | --- |
    | 256 | 256 | 1 | 1 | 65.21GB | 69864MiB |
    | 256 | 128 | 2 | 1 | 33.68GB | 36654MiB |
    | 256 | 64 | 4 | 1 | 17.71GB | 20152MiB |
    | 256 | 32 | 8 | 1 | 9.7GB | 12344MiB |
    | 256 | 16 | 16 | 1 | 5.76GB | 8018MiB |
    | 256 | 8 | 32 | 1 | 3.72GB | 5872MiB |
- Another way to train with a large batch size is to use data parallel strategy, which make model replicas and data batch slices across DP ranks to alleviate the memory consumed per GPU.
    
    
    | global batch size | n_gpus | MaxMemAllocated (CUDA) | Per GPU Mem Usage (NVIDIA-SMI) |
    | --- | --- | --- | --- |
    | 256 | 1 | 65.21GB | 69864MiB |
    | 256 | 2 | 33.25GB | 37038MiB |
    | 256 | 4 | 17.28GB | 21160MiB |
    | 256 | 8 | 9.28GB | 12616MiB |
- It’s also possible to further reduce the memory usage. For example, you can use FP16 data types for training efficiently at the cost of lower precision. Furthermore, deepspeed’s zero optimizations can also be used alone / together in distributed training. Note that for small models, higher zero stages may not help. For most cases, zero1 (optimizer states partitioning) is enough.
    
    
    | global batch size | n_gpus | use fp16 | use zero1 | MaxMemAllocated (CUDA) | Per GPU Mem Usage (NVIDIA-SMI) |
    | --- | --- | --- | --- | --- | --- |
    | 256 | 1 | False | False | 65.21GB | 69864MiB |
    | 256 | 1 | True | False | 35.18GB | 38594MiB |
    | 256 | 8 | False | False | 9.28GB | 12616MiB |
    | 256 | 8 | True | False | 5.71GB | 9066MiB |
    | 256 | 8 | False | True | 8.59GB | 12172MiB |
    | 256 | 8 | True | True | 4.64GB | 7610MiB |
- There are other strategies to maximize hardware usage to gain a better performance. Check [https://www.deepspeed.ai/](https://www.deepspeed.ai/) for more details.

## 5. Inference

- Edit deepspeed_config.json according to your config params 
```bash
bash scripts/predict.sh
```

## Reference

[1] Omelianchuk, K., Atrasevych, V., Chernodub, A., & Skurzhanskyi, O. (2020). GECToR – Grammatical Error Correction: Tag, Not Rewrite. arXiv:2005.12592 [cs]. http://arxiv.org/abs/2005.12592