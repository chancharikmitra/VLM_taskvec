#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python3 -m mtv_eval \
    --model_name Qwen \
    --data_name okvqa \
    --train_path /home/zhaobin/Qwen-VL/data/okvqa/okvqa_train.jsonl \
    --val_path /home/zhaobin/Qwen-VL/data/okvqa/okvqa_val.jsonl \
    --num_example 1 \
    --num_shot 4 \
    --max_token 10 \
    --bernoullis_path /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/okvqa/Bernoullis/test.pt \
    --activation_path /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/okvqa/Bernoullis/test_activation.pt \
    --is_eval True \
    --result_folder /home/zhaobin/Qwen-VL/qwen_reinforce/dataset/okvqa/results/ \
    --cur_mode interv \
    --experiment_name test