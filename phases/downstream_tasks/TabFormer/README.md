# Description

This code base follows the [official repo](https://github.com/IBM/TabFormer) of
paper [Tabular Transformers for Modeling Multivariate Time Series](http://arxiv.org/abs/2011.01843 ).

## Datasets

### Credit Card Transaction Dataset

The synthetic credit card transaction dataset can be downloaded through the
[direct link](https://ibm.box.com/v/tabformer-data).


## Usage

### Pre-training

All the following commands should be run under `TUNA/`.

For TabFormer baseline, please run

```shell
python -m torch.distributed.launch --nproc_per_node [gpu number per node] --use_env \
        -m phases.downstream_tasks.TabFormer.main --field_ce --lm_type bert --field_hs 64 --data_type 'card' \
        --cached --output_dir [output_dir] --mlm --num_train_epochs 3 --stride 5 \
        --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --data_root [data_root]
```

For TUNA + TabFormer, please run

```shell
python -m torch.distributed.launch --nproc_per_node [gpu number per node] --use_env \
        -m phases.downstream_tasks.TabFormer.main --field_ce --lm_type bert --field_hs 64 --data_type 'card' \
        --cached --data_extension 'use_numtok' --use_numtok --number_model_config 'CharLSTM_9M_base_for_tabformer' \
        --output_dir [output_dir] --mlm --num_train_epochs 3 --stride 10 \
        --per_device_train_batch_size 24 --per_device_eval_batch_size 24 --data_root [data_root]
```

If you want to use other number model config, please view `TUNA/number_encoder/config.py`. If you want to run the
control experiment on *Replace* version of numtok, please add `--use_replace`. If you want to add regression loss in
model training, please add `--use_reg_loss`.

For more command options and argument details, please refer to [args.py](./args.py).

### Downstream Tasks

#### Fraud Detection Task

We use TabBERT as the feature extractor and freeze the TabBERT network during LSTM network training.

To extract features using trained model, please run

```shell
python -m torch.distributed.launch --nproc_per_node [gpu number per node] --use_env \
        -m phases.downstream_tasks.TabFormer.extract_features --field_ce --lm_type bert --field_hs 64 \
        --cached --data_type 'card' --data_extension 'use_numtok' --data_root [data_root] \
        --use_numtok --number_model_config 'CharLSTM_9M_base_for_tabformer' \
        --output_dir [output_dir] --mlm --stride 10 --per_device_eval_batch_size 128 \
        --model_dir [model_dir] --cached_feature_dir [cached_feature_dir]
```

To train LSTM network and calculate fraud detection f1 score, please run

```shell
python -m phases.downstream_tasks.TabFormer.fraud_detect \
        --cached_feature_dir [cached_feature_dir] --upsample \
        --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --num_train_epochs 10
```

For more information on task specific arguments, please see [fraud_detect.py](./fraud_detect.py).
