# Description 

This folder contains codes for the downstream task TAT-QA.

# Data Preparation
run:
```
python -m phases.downstream_tasks.TAT.prepare_dataset
```
'ori', 'num' and 'both' respectively denotes the numtok strategy -- "baseline", "replace" and "addback".

Or directly use the prepared data in "data/TAT/*.pkl"


# Experiment
to run roberta:
```
bash phases/downstream_tasks/TAT/scripts/tatqa_roberta.sh
```


for details, run:
```
python phases/downstream_tasks/TAT/runner.py --help
```
