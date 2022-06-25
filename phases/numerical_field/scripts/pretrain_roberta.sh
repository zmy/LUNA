#!/usr/bin/env bash
export EXP_NAME=pretrain_roberta
export GIT_BRANCH=main
#WANDB config
export WANDB_API_KEY=<your wandb key>
export WANDB_PROJECT=LUNA
export WANDB_ENTITY=<your wandb entity>
export WANDB_RUN_GROUP=Pretrain

export MKL_SERVICE_FORCE_INTEL=1

export OUTPUT_DIR_PREFIX=/ckpt/pretrain/${EXP_NAME}
#the output_dir_prefix, the output_dir is usually named OUTPUT_DIR_PREFIX_{seed}
export COMMIT_HASH=`git log -n1 --format=format:"%H"`
#record the commit hash
SEED=42

python -m torch.distributed.launch \
--nproc_per_node=8 \
--node_rank=${NODE_RANK} \
--nnodes=2 \
--master_addr=${MASTER_IP} \
--master_port=${MASTER_PORT} \
--use_env phases/numerical_field/Pretrain.py \
--config phases/numerical_field/configs/small/Pretrain_roberta_16k.yaml \
--keep_origin 1 \
--seed ${SEED} \
--output_dir ${OUTPUT_DIR_PREFIX}_${SEED}
