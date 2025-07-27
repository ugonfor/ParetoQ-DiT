# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# --input_model_filename "Freepik/flux.1-lite-8B" \

# bash 1_run_train.sh 1 4 1
torchrun --nnodes=1 --nproc_per_node=$1 train.py \
--local_dir "/tmp/flux/" \
--input_model_filename "black-forest-labs/FLUX.1-dev" \
--output_model_filename "flux-dev" \
--do_train True \
--do_eval True \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir /tmp/output/runs/current \
--num_train_epochs 1 \
--per_device_train_batch_size $2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps $3 \
--evaluation_strategy "no" \
--eval_steps 500 \
--save_strategy "steps" \
--save_steps 500 \
--report_to "tensorboard" \
--save_total_limit 5 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing True \
--qat True \
--w_bits 0 \
--ddp_find_unused_parameters True \
--optim "adamw_bnb_8bit" \
# --resume_from_checkpoint <path/to/checkpoint> \
# --load_best_model_at_end True \
# --metric_for_best_model "eval_loss" \
# --greater_is_better False