#!/bin/bash
export CUDA_VISIBLE_DEVICES=

python3 -m task_vector_cma \
    --model_path Qwen/Qwen-VL \
    --data_name vizwiz \
    --train_path /home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_train.jsonl \
    --val_path /home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_val.jsonl \
    --num_example 100 \
    --num_shot 4 \
    --is_eval True \
    --cur_mode both \
    --experiment_name vizwiz_cma