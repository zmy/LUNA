# Data Preparation
run:
```
python -m phases.downstream_tasks.TAT.prepare_dataset
```

# Experiment
to run roberta:
```angular2html
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    --use_env phases/downstream_tasks/TAT/runner.py \
    --seed $seed \
    --encoder roberta \
    --use_numtok 1 (or 0)\
    --weights_path $pretrain_ckpt_file_of_phase1 \
    --save_dir $your_path_to_save_results
```
to run tapas:
```angular2html
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    --use_env phases/downstream_tasks/TAT/runner.py \
    --seed $seed \
    --encoder tapas \
    --use_numtok 1 (or 0) \
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

To run on gcr, please add arguments: 
```
--input_dir data/DownstreamDataset/TAT
--model_dir data/ckpt
--redirect_huggingface_cache 0
--tblog_dir auto (to avoid streammingly write blob container)
```
To run on 8 gpus: if you add **"--gradient_accumulation_steps 1"** argument, you will reproduce the result on 4 gpus, but faster. Since logically the same batchsize and steps are used. 
Otherwise, the default accumulation "2" makes the batchsize twice as large as before and num of steps a half of before. 