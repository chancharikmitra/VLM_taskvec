#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

python3 -m mtv_eval \
    --model_name ViLA \
    --data_name vizwiz \
    --train_path /home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_train.jsonl \
    --val_path /home/zhaobin/Qwen-VL/data/vizwiz/vizwiz_val.jsonl \
    --num_example 1 \
    --num_shot 4 \
    --max_token 10 \
    --bernoullis_path /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/vizwiz/Bernoullis/test.pt \
    --activation_path /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/vizwiz/Bernoullis/test_activation.pt \
    --is_eval True \
    --result_folder /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/vizwiz/results/ \
    --cur_mode interv \
    --experiment_name test