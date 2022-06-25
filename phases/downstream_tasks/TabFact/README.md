# Description 

This folder contains codes for the downstream task TabFact.

#Data
TabFact dataset is automatically downloaded when running in python:
```
load_dataset('tab_fact', 'tab_fact', split='train')
```


# Usage
1. Prepare the intermediate pre-trained checkpoint ${MODEL_PATH}


2. As a demo to use CharLSTM on a pretrained bert checkpoint, in `TUNA/` run:

```
bash phases/downstream_tasks/TabFact/scripts/tabfact_bert.sh
```