#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0

OUTPUT_PATH=./output
mkdir -p $OUTPUT_PATH

deepspeed training/step1_supervised_finetuning/main.py \
   --data_path /home/yaohuayang/baoyue.shen/CODE/DeepSpeed/Datasets/Dahoas/rm-static /home/yaohuayang/baoyue.shen/CODE/DeepSpeed/Datasets/Dahoas/full-hh-rlhf /home/yaohuayang/baoyue.shen/CODE/DeepSpeed/Datasets/Dahoas/synthetic-instruct-gptj-pairwise /home/yaohuayang/baoyue.shen/CODE/DeepSpeed/Datasets/yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path /home/yaohuayang/baoyue.shen/CODE/DeepSpeed/opt-1.3B \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 1e-3 \
   --weight_decay 0.1 \
   --num_train_epochs 16 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 0 \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --only_optimize_lora \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
   --data_output_path output\\date_file \
   &> $OUTPUT_PATH/training.log
