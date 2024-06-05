#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

python3 -m mtv_eval \
    --model_name idefics2 \
    --data_name ai2d \
    --train_path  /home/chancharikm/taskvec/VLM_taskvec/data/ai2diagram/train.jsonl \
    --val_path  /home/chancharikm/taskvec/VLM_taskvec/data/ai2diagram/test.jsonl \
    --num_example 100 \
    --num_shot 4 \
    --max_token 20 \
    --is_eval True \
    --bernoullis_path None \
    --activation_path None \
    --cur_mode clean \
    --experiment_name zero