#!/usr/bin/env bash
export EXP_NAME=tabfact_bert
export GIT_BRANCH=main
# branch of your git repo, a "git clone" will be run remotely.

# WANDB config
export WANDB_API_KEY=<your wandb key>
export WANDB_PROJECT=LUNA
export WANDB_ENTITY=<your wandb entity>
export WANDB_RUN_GROUP=TabFact

export MKL_SERVICE_FORCE_INTEL=1

export MODEL_PATH=/ckpt/pretrain/pretrain_bert_42/checkpoint_03.pth
export OUTPUT_DIR_PREFIX=/ckpt/tabfact/${EXP_NAME}
# the output_dir_prefix, the output_dir is usually named OUTPUT_DIR_PREFIX_{seed}

python -m torch.distributed.launch \
--nproc_per_node=8 \
--master_addr=${MASTER_IP} \
--master_port=${MASTER_PORT} \
--use_env \
-m phases.downstream_tasks.TabFact.finetune_tabfact \
--encoder bert \
--weights_path ${MODEL_PATH} \
--use_numtok 2 \
--output_dir ${OUTPUT_DIR} \
--seed $SEED