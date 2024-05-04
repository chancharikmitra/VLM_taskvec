#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

python3 -m task_vector_reinforce_eval \
    --model_path Qwen/Qwen-VL \
    --data_name vizwiz \
    --train_path /home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_train.jsonl \
    --val_path /home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_val.jsonl \
    --num_example 100 \
    --num_shot 4 \
    --num_reinforce 100 \
    --bernoullis_path /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/vizwiz/Bernoullis/theta3.pt \
    --is_eval True \
    --result_folder /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/vizwiz/results/ \
    --cur_mode interv 