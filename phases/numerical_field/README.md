# Data
Coming soon

you can collect the dataset for intermediate pre-training by running: 
```
python -m phases.numerical_field.preprocessing.combine
```
Or directly use the prepared data in "data/PretrainDataset/small/*.json"

## Runing
E.g., to pre-training RoBERTa on two nodes, please follow the steps: 

on worker 0, run:
```
NODE_RANK=0 bash phases/numerical_field/scripts/pretrain_roberta.sh
```

on worker 1, run:
```
NODE_RANK=1 bash phases/numerical_field/scripts/pretrain_roberta.sh
```
