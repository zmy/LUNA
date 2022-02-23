import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel


class RoBERTaEmbedding(nn.Module):

    def __init__(self):
        super(RoBERTaEmbedding, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.roberta = RobertaModel.from_pretrained('roberta-large')

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        output = self.roberta(input_ids=input_ids,
                              attention_mask=attention_mask, return_dict=True)
        last_hidden = output['last_hidden_state']
        masked = last_hidden * attention_mask.unsqueeze(-1)
        lens = torch.sum(attention_mask, dim=1)
        avg_pooled = torch.sum(masked, dim=1) / lens.view(-1, 1)

        return avg_pooled
