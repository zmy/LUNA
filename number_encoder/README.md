# Description
---

The folder `number_encoder/` contains codes that facilitate the process of **num**ber em**bed**ding (numbed). 

`config.py` defines the necessary arguments used by the NumBed class.

`numbed.py` defines the NumBed class itself, which utilizes a variety of embedding backbones defined in `embed_backbone/`.

`embed_backbone/` stores specific implementations of the diffrent backbone models.


# Usage 
---

Use the following Python code block for a quick demo: 

```python 
import torch
from number_encoder import NumBedConfig, NumBed
from number_tokenizer.numtok import NumTok

model_name = 'CharLSTM_base'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_config = NumBedConfig(model_name=model_name)
model = NumBed(model_config).to(device)

texts = ['hello', 'stm32', '1,234,567.890%', '30 June 2018', '1+2-3+4-5']
for text in texts:
    numbers_and_indices = NumTok.find_numbers(text)
    tokenized = NumTok.tokenize(numbers_and_indices, device)
    number_emb = model(tokenized)
    print('"{}"'.format(text), 'has the embedding size', number_emb.shape)
```

