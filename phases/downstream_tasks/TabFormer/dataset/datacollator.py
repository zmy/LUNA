from typing import Dict, List, Tuple, Union

import torch
from number_tokenizer import NumTok
from transformers import DataCollatorForLanguageModeling


class TransDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = torch.stack([x for x in examples])
        sz = batch.shape
        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs.view(sz), "masked_lm_labels": labels.view(sz)}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove "
                "the --mlm flag if you want to use this tokenizer. "
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability
        # defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def numtok_collate_fn(data):
    """
    Stacks batch data and tokenizes raw numbers using NumTok.
    """
    try:
        # For data without labels.
        input_ids = torch.stack([x['input_ids'] for x in data])
        number_lists = []
        for x in data:
            number_lists.extend(x['raw_num'])
        tokenized_numbers = [NumTok.find_numbers(x)[0] for l in number_lists for x in l]
        number_values = torch.tensor([NumTok.get_val(x[0]) for x in tokenized_numbers], dtype=torch.float32)
        tokenized_numbers = NumTok.tokenize(tokenized_numbers)
        tokenized_numbers['number_values'] = number_values

        return {
            'input_ids': input_ids,
            'tokenized_numbers': tokenized_numbers
        }

    except Exception as e:
        # For data with labels.
        input_ids = torch.stack([x[0]['input_ids'] for x in data])
        number_lists = []
        for x in data:
            number_lists.extend(x[0]['raw_num'])
        tokenized_numbers = [NumTok.find_numbers(x)[0] for l in number_lists for x in l]
        tokenized_numbers = NumTok.tokenize(tokenized_numbers)

        labels = torch.stack([x[1] for x in data])

        return {
            'input_ids': input_ids,
            'tokenized_numbers': tokenized_numbers,
            'labels': labels
        }
