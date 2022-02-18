# Description 

This folder contains codes for the downstream task TabFact.

# Usage
1. Prepare the intermediate pre-trained checkpoint (path referred to as `path/to/your/checkpoint`) 


2. As a demo to use TransPos on a pretrained TaPas checkpoint, in `TUNA/` run:

```bash
export CKPT_PATH=<path/to/your/checkpoint>
export ENCODER=tapas
export MODEL=TransPos
export SEED=42

# Prepare TabFact data and fine-tune the pretrained checkpoint on TabFact data
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m phases.downstream_tasks.TabFact.finetune_tabfact \
--batch_size 48 \
--encoder $ENCODER \
--max_epoch 10 \
--seed $SEED \
--weights_path /storage/tuna-models/$CKPT_PATH \
--numbed_model_name TransPos \
--output_dir /storage/tuna-models/tabfact-ckpt/${MODEL}_${SEED}/

# Evaluate on TabFact develop set
export CUDA_VISIBLE_DEVICES=0
for i in 9 8 7 6 5 4 3 2 1 0
do
    python -m phases.downstream_tasks.TabFact.finetune_tabfact \
    --batch_size 12 \
    --ckpt /storage/tuna-models/tabfact-ckpt/${MODEL}_${SEED}/net_$i.pt \
    --encoder $ENCODER \
    --mode valid \
    --numbed_model_name ${MODEL}
    echo "net_$i"
done

# Evaluate on TabFact test set
python -m phases.downstream_tasks.TabFact.finetune_tabfact \
--batch_size 12 \
--ckpt /storage/tuna-models/tabfact-ckpt/${MODEL}_${SEED}/net_$best.pt \
--encoder $ENCODER \
--mode test \
--numbed_model_name ${MODEL}