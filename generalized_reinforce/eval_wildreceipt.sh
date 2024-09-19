#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

python3 -m mtv_eval \
    --model_name llava-onevision \
    --data_name wildreceipt \
    --train_path ../data/wildreceipt/test.json \
    --val_path ../data/wildreceipt/test.json \
    --num_example 2 \
    --num_shot 8 \
    --max_token 10 \
    --bernoullis_path ../data/wildreceipt/Bernoullis/MTV_nonum_8.pt \
    --activation_path ../data/wildreceipt/Bernoullis/MTV_nonum_8_activation.pt \
    --is_eval True \
    --result_folder ../data/wildreceipt/results/ \
    --cur_mode clean \
    --experiment_name MTV_nonum_8
