#!/usr/bin/env bash
export EXP_NAME=tabformer
export GIT_BRANCH=main
export MODEL_CONFIG=CharLSTM_9M_base_for_tabformer


#WANDB config
export WANDB_API_KEY=<your wandb key>
export WANDB_PROJECT=LUNA
export WANDB_ENTITY=<your wandb entity>
export WANDB_RUN_GROUP=TabBERTPretrain

export MKL_SERVICE_FORCE_INTEL=1

export OUTPUT_DIR_PREFIX=/ckpt/creditrans/${EXP_NAME}
#the output_dir_prefix, the output_dir is usually named OUTPUT_DIR_PREFIX_{seed}
export COMMIT_HASH=`git log -n1 --format=format:"%H"`
#record the commit hash


# TabFormer pre-training.
python -m torch.distributed.launch --use_env --nproc_per_node=8 \
        -m phases.downstream_tasks.TabFormer.main --field_ce --lm_type bert --field_hs 64 --data_type 'card' \
        --cached --use_numtok --number_model_config ${MODEL_CONFIG} \
        --data_extension 'use_numtok' \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR_PREFIX}'/'${MODEL_CONFIG}'_seed'${SEED} --mlm --num_train_epochs 1 --stride 10 \
        --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --data_root './data/PretrainDataset/tabformer/credit_card'

# TabFormer feature extraction.
# We only need to extract feature once and run different seeds on these extracted features because there is no
# randomness in the feature extraction stage.
python -m torch.distributed.launch --use_env --nproc_per_node=2 \
        -m phases.downstream_tasks.TabFormer.extract_features --field_ce --lm_type bert --field_hs 64 --data_type 'card' \
        --cached --use_numtok --number_model_config ${MODEL_CONFIG} \
        --data_extension 'use_numtok' \
        --model_name_or_path ${OUTPUT_DIR_PREFIX}'/'${MODEL_CONFIG}'_seed'${SEED} \
        --output_dir ${OUTPUT_DIR_PREFIX}'/'${MODEL_CONFIG}'_seed'${SEED} --mlm  --stride 10 \
        --per_device_eval_batch_size 128 --data_root './data/PretrainDataset/tabformer/credit_card' --cached_feature_dir './tabformer_features'

export WANDB_RUN_GROUP=TabBERTFraudDetect
# Fraud Detection task.
seeds=(2022)
for round in "${!seeds[@]}";
do
  export WANDB_RUN_NOTES="running ${EXP_NAME} with seed ${seed[$round]}"
  python -m phases.downstream_tasks.TabFormer.fraud_detect \
          --seed ${seed[$round]} \
          --cached_feature_dir './tabformer_features' --upsample \
          --use_numtok --number_model_config ${MODEL_CONFIG} \
          --model_name_or_path ${OUTPUT_DIR_PREFIX}'/'${MODEL_CONFIG}'_seed'${SEED} \
          --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --num_train_epochs 10
done
