# Description 
--- 

This folder contains codes for the fine-tuning and evaluation process on TabFact. We test on TaPas. 

# Usage
--- 
1. Prepare the pretrained checkpoint (path referred to as `path/to/your/checkpoint`) 


2. As a demo to use TransPos on a pretrained TaPas checkpoint, in `TUNA/` run:

```bash
export CKPT_PATH=<path/to/your/checkpoint>
export ENCODER=tapas
export MODEL=TransPos
export SEED=42

# Prepare TabFact data and fine-tune the pretrained checkpoint on TabFact data
for CUDA_VISIBLE_DEVICES in 0 0,1,2,3
do
python -m phases.downstream_tasks.TabFact.finetune_tabfact \
--batch_size 48 \
--encoder $ENCODER \
--max_epoch 10 \
--seed $SEED \
--weights_path /storage/tuna-models/$CKPT_PATH \
--numbed_model_name TransPos \
--output_dir /storage/tuna-models/tabfact-ckpt/${MODEL}_${SEED}/
done

# Evaluate on TabFact testset
export CUDA_VISIBLE_DEVICES=0
for i in 9 8 7 6 5 4 3 2 1 0
do
    python -m phases.downstream_tasks.TabFact.finetune_tabfact \
    --batch_size 12 \
    --ckpt /storage/tuna-models/tabfact-ckpt/${MODEL}_${SEED}/net_$i.pt \
    --encoder $ENCODER \
    --mode test \
    --numbed_model_name ${MODEL}
    echo "net_$i"
done

```

To run on gcr, please add arguments:
```
--data_dir data/DownstreamDataset/huggingface_TabFact
--model_dir data/ckpt
--redirect_huggingface_cache 0
```

