#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

python3 -m mtv_eval \
    --model_name llava-onevision \
    --data_name kvp \
    --train_path ../data/KVP10k_data/test.json \
    --val_path ../data/KVP10k_data/test.json \
    --num_example 1 \
    --num_shot 8 \
    --max_token 10 \
    --bernoullis_path ../data/KVP10k_data/Bernoullis/MTV_8.pt \
    --activation_path ../data/KVP10k_data/Bernoullis/MTV_8_activation.pt \
    --is_eval True \
    --result_folder ../data/KVP10k_data/results/ \
    --cur_mode clean \
    --experiment_name MTV_8
