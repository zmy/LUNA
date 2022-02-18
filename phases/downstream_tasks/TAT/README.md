# Description 

This folder contains codes for the downstream task TAT-QA.

# Data Preparation
run:
```
python -m phases.downstream_tasks.TAT.prepare_dataset
```
'ori', 'num' and 'both' respectively denotes the numtok strategy -- "baseline", "replace" and "append". 
# Experiment
to run roberta:
```angular2html
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    --use_env phases/downstream_tasks/TAT/runner.py \
    --seed $seed \
    --encoder roberta \
    --use_numtok 2 (or 0,1)\
    --weights_path $pretrain_ckpt_file_of_phase1 \
    --save_dir $your_path_to_save_results
```
to run tapas:
```angular2html
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    --use_env phases/downstream_tasks/TAT/runner.py \
    --seed $seed \
    --encoder tapas \
    --use_numtok 2 (or 0,1) \
    --learning_rate 3e-4 \
    --bert_learning_rate 1e-5 \
    --batch_size_per_node 4 \
    --eval_batch_size_per_node 4 \
    --weights_path $pretrain_ckpt_file_of_phase1 \
    --save_dir $your_path_to_save_results
```

for details, run:
```angular2html
python phases/downstream_tasks/TAT/runner.py --help
```
