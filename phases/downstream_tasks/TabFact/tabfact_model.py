import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TapasForSequenceClassification, RobertaForSequenceClassification, BertForSequenceClassification
import os
from number_encoder.numbed import NumBed, NumBedConfig
from number_tokenizer.numtok import NumTok
from number_encoder.sembed import SemBed


class TabFactModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.use_sembed = args.model_name.split('_')[0] == 'SemLit'
        cache_dir = os.path.join(args.model_dir,
                                 'huggingface') if args.redirect_huggingface_cache else None
        if args.encoder == 'tapas':
            self.encoder = TapasForSequenceClassification.from_pretrained("google/tapas-base-masklm",
                                                                          cache_dir=cache_dir)
            if self.use_sembed:
                self.sembed = SemBed('TaPas')
        elif args.encoder == 'roberta':
            self.encoder = RobertaForSequenceClassification.from_pretrained(
                os.path.join(args.model_dir, 'roberta.large'))
            if self.use_sembed:
                self.sembed = SemBed('RoBERTa')
        elif args.encoder == 'bert':
            self.encoder = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                         cache_dir=cache_dir)
            if self.use_sembed:
                self.sembed = SemBed('BERT')
        self.encoder.resize_token_embeddings(len(tokenizer))
        self.use_numbed = args.model_name != 'zero'
        if self.use_numbed:
            number_model_config = NumBedConfig(model_name=args.model_name,
                                               encoder_name='RoBERTa' if args.encoder == 'roberta' else 'TaPas',
                                               checkpoint_path=args.checkpoint_path)
            self.number_model = NumBed(number_model_config)
        self.h_dim = 768 if args.encoder in {'tapas', 'bert'} else 1024
        self.backbone = args.encoder
        if args.kept_keys == '':
            self.kept_keys = ()
        else:
            self.kept_keys = args.kept_keys.split(',')

    def forward(self, input_ids, attention_mask, label, number_list, token_type_ids=None, **kwargs):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        text_embedding = getattr(self.encoder, self.backbone).embeddings.word_embeddings(input_ids)
        input_embedding = text_embedding
        if self.use_numbed:
            batch_number_emb = []
            for _number_list in number_list:
                if _number_list == '':
                    continue
                else:
                    numbers = [(x, None, None) for x in _number_list.split('#')]
                    token = NumTok.tokenize(numbers, input_ids.device, self.kept_keys)
                    semantics = self.sembed(_number_list.split('#'), input_ids.device,
                                            self.encoder) if self.use_sembed else {}
                    embed = self.number_model({**token, **semantics})
                batch_number_emb.append(embed)
            if len(batch_number_emb) > 0:
                _num_embedding = torch.cat(batch_number_emb, axis=0)
            else:
                _num_embedding = torch.zeros(0, self.h_dim).to(input_ids.device)

            num_embedding = torch.cat((torch.zeros((1, self.h_dim)).to(_num_embedding), _num_embedding), axis=0)
            num_id = torch.zeros_like(input_ids)
            num_indice = input_ids == self.tokenizer.num_token_id
            if _num_embedding.size(0) > 0:
                num_id[num_indice] = torch.range(1, _num_embedding.size(0)).to(input_ids)
            num_embedding = F.embedding(num_id, num_embedding)
            input_embedding = input_embedding + num_embedding

        # forward pass
        outputs = self.encoder(inputs_embeds=input_embedding, attention_mask=attention_mask,
                               token_type_ids=token_type_ids, return_dict=True,
                               labels=label)

        loss = outputs.loss

        pred = outputs.logits.argmax(-1)
        acc = (pred == label).sum() / (label.size(0) + 1e-7)
        return loss, acc, pred
