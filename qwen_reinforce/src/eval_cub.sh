#!/bin/bash
export CUDA_VISIBLE_DEVICES=5,6,7

python3 -m task_vector_reinforce_eval \
    --model_path Qwen/Qwen-VL \
    --data_name cub \
    --train_path /home/zhaobin/Qwen-VL/task_vector/cub/cub_train.json \
    --val_path /home/zhaobin/Qwen-VL/task_vector/cub/cub_test.json \
    --num_example 100 \
    --num_shot 0 \
    --num_reinforce 100 \
    --bernoullis_path /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/cub/Bernoullis/theta_100avg.pt \
    --is_eval True \
    --result_folder /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/cub/results/ \
    --cur_mode both \
    --experiment_name RL_100avg