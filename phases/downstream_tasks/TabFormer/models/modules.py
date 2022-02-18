from transformers import (
    BertTokenizer,
    BertForMaskedLM,
)
from transformers.modeling_utils import PreTrainedModel

from .hierarchical import TabFormerEmbeddings, NumNetTabFormerEmbeddings
from .tabformer_bert import TabFormerBertForMaskedLM, TabFormerBertConfig


class TabFormerBaseModel(PreTrainedModel):
    def __init__(self, hf_model, tab_embeddings, config):
        super().__init__(config)

        self.model = hf_model
        self.tab_embeddings = tab_embeddings

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerHierarchicalLM(PreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = TabFormerBertForMaskedLM(self.config, vocab)

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class NumNetTabFormerHierarchicalLM(PreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab, number_model_config, tokenizer, mlm_probability=0.15):
        super().__init__(config)

        self.config = config
        self.number_model_config = number_model_config
        self.mlm_probability = mlm_probability

        self.tab_embeddings = NumNetTabFormerEmbeddings(self.config, number_model_config, tokenizer, mlm_probability)
        self.tb_model = TabFormerBertForMaskedLM(self.config, vocab)

    def forward(self, input_ids, tokenized_number, mlm, **input_args):
        if mlm:
            inputs_embeds, masked_lm_labels = self.tab_embeddings(input_ids=input_ids,
                                                                  tokenized_number=tokenized_number, mlm=mlm)
            return self.tb_model(inputs_embeds=inputs_embeds, masked_lm_labels=masked_lm_labels,
                                 number_values=tokenized_number['number_values'], **input_args)
        else:
            inputs_embeds = self.tab_embeddings(input_ids=input_ids, tokenized_number=tokenized_number, mlm=mlm)
            return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerBertLM:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=False, ncols=None, field_hidden_size=768,
                 use_numtok=False, number_model_config=None, use_replace=False, use_reg_loss=False, data_type=None):

        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        hidden_size = field_hidden_size if flatten else (field_hidden_size * self.ncols)

        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=self.ncols,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          flatten=flatten,
                                          num_attention_heads=self.ncols,
                                          use_numtok=use_numtok,
                                          use_replace=use_replace,
                                          use_reg_loss=use_reg_loss,
                                          data_type=data_type)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model(field_ce, flatten, use_numtok, number_model_config, self.tokenizer)

    def get_model(self, field_ce, flatten, use_numtok, number_model_config=None, tokenizer=None):

        if flatten and not field_ce:
            # flattened vanilla BERT
            model = BertForMaskedLM(self.config)
        elif flatten and field_ce:
            # flattened field CE BERT
            model = TabFormerBertForMaskedLM(self.config, self.vocab)
        else:
            # hierarchical field CE BERT
            if use_numtok:
                model = NumNetTabFormerHierarchicalLM(self.config, self.vocab, number_model_config, tokenizer)
            else:
                model = TabFormerHierarchicalLM(self.config, self.vocab)

        return model
