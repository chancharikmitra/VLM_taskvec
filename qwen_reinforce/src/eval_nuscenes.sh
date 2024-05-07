#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4

python3 -m task_vector_reinforce_eval \
    --model_path Qwen/Qwen-VL \
    --data_name nuscenes \
    --train_path /home/zhaobin/MileBench/data/nuscenes.json \
    --val_path /home/zhaobin/MileBench/data/nuscenes.json \
    --num_example 5 \
    --num_shot 1 \
    --num_reinforce 10 \
    --bernoullis_path /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/nuscenes/Bernoullis/theta_600iter.pt \
    --is_eval True \
    --result_folder /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/nuscenes/results/ \
    --cur_mode interv \
    --experiment_name RL_nuscenes