# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

torchrun --nnodes=1 --nproc_per_node=1 train.py \
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
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "steps" \
--eval_steps 100 \
--save_strategy "steps" \
--save_steps 400 \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0. \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--gradient_checkpointing False \
--qat True \
--w_bits 0 \