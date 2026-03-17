#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2

# Run training
python train_rap_ldp.py \
    --model_name bert-base-uncased \
    --task_name sst2 \
    --num_epochs 3 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_seq_length 128 \
    --num_blocks 12 \
    --rl_interval 100 \
    --sac_start_steps 200 \
    --sac_buffer_size 10000 \
    --sac_batch_size 64 \
    --sac_updates_per_interval 6 \
    --actor_lr 2e-5 \
    --critic_lr 1e-4 \
    --alpha 0.2 \
    --gamma 0.99 \
    --tau 0.005 \
    --epsilon 8.0 \
    --delta 1e-5 \
    --default_norm_c 1.0 \
    --eval_batches_for_state 20 \
    --output_dir ./rap_ldp_output_bert_sst2_best \
    --seed 42 \
    --use_data_parallel \
    --use_wandb \
    --wandb_project RAP-LDP \
    --wandb_run_name bert_sst2_best_reproduce \
    --reward_mode soft_tanh
