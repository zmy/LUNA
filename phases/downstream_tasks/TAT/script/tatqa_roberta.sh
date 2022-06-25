#!/usr/bin/env bash
export EXP_NAME=tatqa_roberta
#name of your cluster, run "amlt target info amlk8s" to select
export GIT_BRANCH=main
export MODEL_PATH=/ckpt/pretrain/pretrain_roberta_42/checkpoint_03.pth


#WANDB config
export WANDB_API_KEY=<your wandb key>
export WANDB_PROJECT=LUNA
export WANDB_ENTITY=<your wandb entity>
export WANDB_RUN_GROUP=TATQA

export MKL_SERVICE_FORCE_INTEL=1

export OUTPUT_DIR_PREFIX=/ckpt/tatqa/${EXP_NAME}
#the output_dir_prefix, the output_dir is usually named OUTPUT_DIR_PREFIX_{seed}
export COMMIT_HASH=`git log -n1 --format=format:"%H"`
#record the commit hash

python -m torch.distributed.launch \
--nproc_per_node=8 \
--master_addr=${MASTER_IP} \
--master_port=${MASTER_PORT} \
--use_env phases/downstream_tasks/TAT/runner.py \
--encoder roberta \
--input_dir data/TAT \
--weights_path ${MODEL_PATH} \
--save_dir ${OUTPUT_DIR_PREFIX}_${SEED} \
--use_numtok 2 \
--model_name CharLSTM_9M_large \
--seed ${SEED} \
--checkpoint_path random \
--model_dir data/ckpt \
--redirect_huggingface_cache 0
