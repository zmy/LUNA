# Data
Coming soon

you can collect the dataset for intermediate pre-training by running: 
```
python -m phases.numerical_field.preprocessing.combine
```


# Running
clone TUNA repos in ~/, and run: 
```angular2html
cd TUNA
ln -s /table2text_data_eus2/pretrain_data/ data
```
## singular
In 'TUNA', run the following code to pretrain roberta:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env phases/numerical_field/Pretrain.py \
--config phases/numerical_field/configs/small/Pretrain_roberta_8k.yaml \
#use Pretrain_tapas_8k.yaml to pretrain tapas
#use Pretrain_bert_8k.yaml to pretrain bert
```

## distributed
To run ddp on multiple nodes, please follow the steps: 

on worker 0, run:
```angular2
worker0:~/TUNA$ hostname -I
192.168.0.38 172.17.0.1 10.46.224.0
```
select the first address as master_address. 

let's take a **2-node** job as an example:

on worker 0, run:
```angular2
python -m torch.distributed.launch --nproc_per_node=8 \
--nnodes=2 --node_rank=0 --master_addr=192.168.0.38 \
--master_port=12480 （change it as you wish） \
--use_env phases/numerical_field/Pretrain.py \
--config phases/numerical_field/configs/small/Pretrain_roberta_16k.yaml \
--keep_origin 1 \
--output_dir data/ckpt/debug
```

on worker 1, run:
```angular2
python -m torch.distributed.launch --nproc_per_node=8 \
--nnodes=2 --node_rank=1 --master_addr=192.168.0.38 \
--master_port=12480 \
--use_env phases/numerical_field/Pretrain.py \
--config phases/numerical_field/configs/small/Pretrain_roberta_16k.yaml \
--keep_origin 1 \
--output_dir data/ckpt/debug
```
the only difference is the "--node_rank"

To run 4-node job, change "--nnode" to 4, change "--config" to "phases/numerical_field/configs/small/Pretrain_roberta_32k.yaml", and extraly run similar command on worker 2 and 3 by setting "--node_rank" to 2 and 3.

