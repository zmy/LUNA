# Description

This anonymous repository contains code for the core models and description for the data used in *LUNA: Language Understanding with Number Augmentations on Transformers via Number Plugins and Pre-training*. 

The folders [number_encoder](number_encoder)  and [number_tokenizer](number_encoder) contain code for **NumBed** and **NumTok** (Sec. 3.1), respectively.

The folder [phases/numerical_field](phases/numerical_field) contains codes for number pre-training (Sec. 3.2).

The folder [phases/single_number](phases/single_number) contains codes for the toy task.

The folder [phases/downstream_tasks](phases/downstream_tasks) contains codes for the downstream tasks (Sec. 4), including [TAT-QA](phases/downstream_tasks/TAT), [TabFact](phases/downstream_tasks/TabFact), and [CrediTrans](phases/downstream_tasks/TabFormer). 

The folder [phases/empirical_study](phases/empirical_study) contains codes for the empirical studies (Sec. E in appendix), including visualization of attention maps and embeddings from different transformer layers, please rename `attention.ipy123nb` to `attention.ipynb` 

# Preparation
unzip the data.zip and put the `data` dir in this repo as `LUNA/data`. 

To build docker image, run: 
```
docker build -t luna:1.0 .
```

To launch the docker container, run:
```
docker run --rm -it --shm-size=8g luna:1.0 /bin/bash
```

To run each experiment, see the `README` document in each directory under [phases](phases) (as mentioned above).