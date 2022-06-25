'''
SemBed takes a number list and outputs
1) The token mapping in a batch
2) The semantic embedding in a batch
e.g. TODO: an example of the way it works
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import BertTokenizer, RobertaTokenizer
from number_tokenizer.numtok import unpack_and_pad_number_property
# from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig
from torch.nn.utils.rnn import pad_sequence


class SemBed(nn.Module):
    def __init__(self, model_name: str = 'TaPas'):
        if model_name in ['TaPas', 'BERT']:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if model_name in ['RoBERTa']:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        if model_name in ['BART']:
            self.bpe = GPT2BPE(GPT2BPEConfig()).bpe
        self.model_name = model_name
        super(SemBed, self).__init__()

    def forward(self, number_list, device, encoder):
        # Get token_mapping and semantic_embedding, model uses different strategies for different
        # encoding models
        result = {}
        batch_token_mapping = unpack_and_pad_number_property(self.get_token_mapping(number_list), device=device)
        result['batch_token_mapping'] = batch_token_mapping
        batch_semantic_embedding = self.get_semantic_embedding(number_list, encoder, device)
        result['batch_semantic_embedding'] = batch_semantic_embedding
        return result

    def get_token_mapping(self, number_list: List[str]):
        # Get token mapping for different models
        batch_token_mapping = []
        # When using Huggingface models
        if self.model_name in ['TaPas', 'BERT', 'RoBERTa']:
            for number in number_list:
                input_ids = self.tokenizer(number)['input_ids']
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[1:-1])
                clean_tokens = [_.strip('##') for _ in tokens]
                single_token_mapping = []
                for token_idx, token in enumerate(clean_tokens):
                    single_token_mapping.extend([str(token_idx+1)]*len(token))
                batch_token_mapping.append('|'.join(single_token_mapping))
            return batch_token_mapping
        # When using BPE from Fairseq
        elif self.model_name in ['BART']:
            for number in number_list:
                input_ids = self.bpe.encode(number)
                clean_tokens = [self.bpe.decode([_]) for _ in input_ids]
                single_token_mapping = []
                for token_idx, token in enumerate(clean_tokens):
                    single_token_mapping.extend([str(token_idx+1)]*len(token))
                batch_token_mapping.append('|'.join(single_token_mapping))
            return batch_token_mapping

    def get_semantic_embedding(self, number_list: List[str], encoder: nn.Module, device):
        """Get the semantic embeddings of the original tokens that appear in the number_list tokenization result

        :param number_list: the number_list from corpus
        :type number_list: List[str]
        :param encoder: the encoder being used by the main model
        :type encoder: nn.Module
        :param device: the device used
        :type device: String
        :return: The semantic embeddings in a batch
        :rtype: torch.Tensor
        """
        if self.model_name in ['TaPas', 'BERT', 'RoBERTa']:
            # If the model is TaPas, BERT or RoBERTa, we get semantic embeddings
            # by querying the encoder embedding weight with the original ids
            assert encoder is not None
            embedding_weight = getattr(encoder, self.model_name.lower()).embeddings.word_embeddings.weight
            batch_original_ids = unpack_and_pad_number_property(self.get_original_ids(number_list), device=device)
            semantic_embedding = F.embedding(batch_original_ids, embedding_weight)
            return semantic_embedding
        elif self.model_name in ['BART']:
            assert encoder is not None
            embedding_weight = encoder.embed_tokens.weight
            batch_original_ids = pad_sequence(
                [torch.tensor(self.bpe.encode(_)).long() for _ in number_list],
                batch_first=True,
                padding_value=0
            ).to(device)
            semantic_embedding = F.embedding(batch_original_ids, embedding_weight)
            return semantic_embedding

    def get_original_ids(self, number_list: List[str]):
        return ['|'.join(list(map(str, self.tokenizer(number)['input_ids']))[1:-1]) for number in number_list]
