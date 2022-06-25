# Paper Experiment shortcut
---
The current checkpoints are in progress. Current effort is mainly on trimming the loss profile.

# Description 
-----
Scripts in this folder finds the best number embedding model by performing evaluations on the following three tasks:
### One-number representation 
1. **sig** regression: regress the significand value
2. **exp** regression: classify the exponent of the number
3. **log** regression: regress the log value
4. **frac**: classify the number of fractional digits
5. **in01**: classify whether the number is in the range [0, 1]
6. **in0100**: classify whether the number is in the range [0, 100]

### Two-number comparison
1. **add_sig**: regress the significand of the sum
2. **add_exp**: classify the exponent of the sum
3. **add_log**: regress the log value of the sum 
4. **subtract_sig**: regress the significand of the difference
5. **subtract_exp**: classify the exponent of the difference
6. **subtract_log**: regress the log value of the sdifference
7. **CP**: regress the length of common prefix
8. **CS**: regress the length

### Multi-number superlative comparison
1. **max**: classify the maximum number in a list of 5

# Usage
-----
## Generate data 
Use `data/generation/BuildNumBedDatasets.ipynb`,
Copy the JSON files to `TUNA/exp/exp_data/`

Or, copying from dlvm08:
```bash
mkdir -p exp/exp_data
```
TODO: to automate the generation process

## Train the NumBed model
Under `TUNA/`, run:

```bash
# export MODEL=CharLSTM_9M
# export MODEL=TutaFeat_9M
# export MODEL=TransPos_9M
# export MODEL=Streal_9M
export MODEL=CharLSTM_1M
# export MODEL=TutaFeat_1M
# export MODEL=TransPos_1M
# export MODEL=Streal_1M
# export MODEL_SIZE=large
export MODEL_SIZE=base
# export ENCODER_NAME=RoBERTa
export ENCODER_NAME=TaPas
python -m phases.single_number.phase0 \
--model_name ${MODEL}_${MODEL_SIZE} \
--encoder_name ${ENCODER_NAME} \
--objs single double multi \
--evals single double multi \
--epoch_num 30 \
--batch_size 256 \
--lr 1e-3 \
--dataset_name sampled \
--listmax_dataset_name sampled \
--seed 42 \
--device 0 \
--exp_root exp \
--tb_root_dir exp/tb_logs
```


### CharLSTM_base
CharLSTM_base_norm: (use layer norm)
```bash
python -m phases.single_number.phase0 \
--model_name CharLSTM \
--model_suffix Bi \
--preprocess_type trivial \
--mode base_norm \
--objs single double multi \
--evals single double multi \
--epoch_num 30 \
--batch_size 256 \
--lr 1e-3 \
--dataset_name sampled \
--listmax_dataset_name sampled \
--seed 42 \
--device 0 \
--exp_root exp \
--emb_size 768 \
--hidden_size 128 \
--lstm_num_layers 1 \
--use_layer_norm
```

CharLSTM_base_align_with_orig: (use alignment)
```bash
python -m phases.single_number.phase0 \
--model_name CharLSTM \
--model_suffix Bi \
--preprocess_type trivial \
--mode base_align_with_orig \
--objs single double multi \
--evals single double multi \
--epoch_num 30 \
--batch_size 256 \
--lr 1e-3 \
--dataset_name sampled \
--listmax_dataset_name sampled \
--seed 42 \
--device 0 \
--exp_root exp \
--emb_size 768 \
--hidden_size 128 \
--lstm_num_layers 1 \
--align_with_orig
```

CharLSTM_base_norm_align_with_orig: (use layer norm and alignment)
```bash
python -m phases.single_number.phase0 \
--model_name CharLSTM \
--model_suffix Bi \
--preprocess_type trivial \
--mode base_norm_align_with_orig \
--objs single double multi \
--evals single double multi \
--epoch_num 30 \
--batch_size 256 \
--lr 1e-3 \
--dataset_name sampled \
--listmax_dataset_name sampled \
--seed 42 \
--device 0 \
--exp_root exp \
--emb_size 768 \
--hidden_size 128 \
--lstm_num_layers 1 \
--use_layer_norm \
--align_with_orig
```


One could tweak `emb_size`, `hidden_size`, and `lstm_num_layers` to adjust the number of parameters of the CharLSTM model. 

## Use a NumBed checkpoint


Use the following Python code block: 

```python 
import torch
from number_encoder import NumBedConfig, NumBed
from number_tokenizer.numtok import NumTok

model_name = 'CharLSTM_1M'
# model_name = 'TutaFeat_1M'
# model_name = 'TransPos_1M'
# model_name = 'Streal_1M'
checkpoint_path = '<path to your checkpoint>'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_config = NumBedConfig(model_name=model_name,
                            checkpoint_path=checkpoint_path)
model = NumBed(model_config).to(device)

texts = ['hello', 'stm32', '1,234,567.890%', '30 June 2018', '1+2-3+4-5']
for text in texts:
    numbers_and_indices = NumTok.find_numbers(text)
    tokenized = NumTok.tokenize(numbers_and_indices, device=device, kept_keys=model.param_keys)
    number_emb = model(tokenized)
    print('"{}"'.format(text), 'has the embedding size', number_emb.shape)
```