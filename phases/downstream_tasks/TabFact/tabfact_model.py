import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TapasForSequenceClassification, RobertaForSequenceClassification, BertForSequenceClassification
import os
from number_encoder import NumBed, NumBedConfig,SelfAttention_forward
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
        self.use_prompt=args.use_numtok==3
        if self.use_numbed:
            self.number_model_config = NumBedConfig(model_name=args.model_name,
                                               encoder_name='RoBERTa' if args.encoder == 'roberta' else 'TaPas',
                                               checkpoint_path=args.checkpoint_path,
                                               prompt_layers=None if not self.use_prompt else (24 if args.encoder == 'roberta' else 12))
            self.number_model = NumBed(self.number_model_config)
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
        origin_forward = []
        if self.use_numbed:
            all_numbers = []
            for _number_list in number_list:
                if _number_list == '':
                    continue
                all_numbers.extend([(x, None, None) for x in _number_list.split('#')])

            token = NumTok.tokenize(all_numbers, input_ids.device, self.kept_keys)
            semantics = self.sembed([x[0] for x in all_numbers], input_ids.device,
                                    self.encoder) if self.use_sembed and len(all_numbers)>0 else {}
            numtok_dict={**token, **semantics}
            if self.use_prompt:
                if len(all_numbers) != 0:
                    attention_mask[input_ids==self.tokenizer.num_token_id]=0
                    batch_pos_id = torch.where(input_ids == self.tokenizer.num_token_id)[1]
                    numtok_dict['batch_pos_embed']=getattr(self.encoder, self.backbone).embeddings.position_embeddings(batch_pos_id)
                    num_prompt,prompt_mask = self.number_model.prompting(input_ids, numtok_dict, self.tokenizer.num_token_id)
                    num_prompt=num_prompt.view(*(num_prompt.size()[:2]),self.number_model_config.prompt_layers,2,-1)
                    num_prompt=num_prompt[:,:self.args.max_prompt_len]
                    prompt_mask=prompt_mask[:,:self.args.max_prompt_len]
                    for i in range(len(getattr(self.encoder, self.backbone).encoder.layer)):
                        _self=getattr(self.encoder, self.backbone).encoder.layer[i].attention.self
                        origin_forward.append(_self.forward)
                        _self.forward = lambda *args,**kwargs: SelfAttention_forward(_self,*args,
                                                                                      number_prompt=num_prompt[:,:,i],
                                                                                      number_prompt_mask=prompt_mask,
                                                                                      **kwargs)
            elif len(all_numbers) != 0:
                input_embedding = input_embedding + self.number_model.embedding(input_ids, numtok_dict, self.tokenizer.num_token_id)

        # forward pass
        outputs = self.encoder(inputs_embeds=input_embedding, attention_mask=attention_mask,
                               token_type_ids=token_type_ids, return_dict=True,
                               labels=label)
        if len(origin_forward)>0:
            for i in range(len(getattr(self.encoder, self.backbone).encoder.layer)):
                _self = getattr(self.encoder, self.backbone).encoder.layer[i].attention.self
                _self.forward = origin_forward[i]

        loss = outputs.loss

        pred = outputs.logits.argmax(-1)
        acc = (pred == label).sum() / (label.size(0) + 1e-7)
        return loss, acc, pred
