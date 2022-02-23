import json
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import parse_string,compute_col_distribution


class TabertDataset(Dataset):
    def __init__(self, model, max_seq_length):
        assert max_seq_length <= 512
        self.model = model
        self.max_seq_length = max_seq_length
        with open('data/PretrainDataset/all_table.json', encoding='utf8') as f:
            self.tables = json.load(f)
        with open('data/PretrainDataset/all_text.json', encoding='utf8') as f:
            self.texts = json.load(f)
        with open('data/PretrainDataset/all_pair.json', encoding='utf8') as f:
            self.pairs = np.array(json.load(f), dtype=np.object)

        self.table_numbers = {}
        self.table_distributions = {}
        self.text_numbers = {}
        self.cache = {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        from ..models.table_bert import Table, Column
        if index in self.cache:
            return self.cache[index]
        table_id, text_id = self.pairs[index]
        table = self.tables[table_id]
        text = self.texts[text_id]
        table = table[:10]
        if len(table) == 0:
            df = Table(
                id='',
                header=[
                    Column('[MASK]', 'text'),
                ],
                data=[
                    ['[MASK]'],
                ]
            ).tokenize(self.model.tokenizer)
        elif len(table) == 1:
            df = Table(
                id='',
                header=[Column(x, 'text') for x in table[0]],
                data=[
                    ['[MASK]'] * len(table[0]),
                ]
            ).tokenize(self.model.tokenizer)
        else:
            df = Table(
                id='',
                header=[Column(x, 'text') for x in table[0]],
                data=table[1:20]
            ).tokenize(self.model.tokenizer)

        col = []
        for j, row in enumerate(df.data_number):
            for i in range(len(row)):
                if j == 0:
                    col.append([])
                if len(row[i]) == 1:
                    col[i].append(row[i][0][1])
                else:
                    col[i].append(None)
        table_distribution = []
        for x in col[:50]:
            table_distribution.append(compute_col_distribution(x))
        while len(table_distribution) < 50:
            table_distribution.append([-1e11] * 5)
        table_distribution = torch.tensor(table_distribution, dtype=torch.float32)
        text, number_list = parse_string(text, 256,7)
        text = self.model.tokenizer.tokenize(text)
        instance = self.model.input_formatter.get_input((text, number_list), df,max_length=self.max_seq_length)
        for row_inst in instance['rows']:
            row_inst['token_ids'] = self.model.tokenizer.convert_tokens_to_ids(row_inst['tokens'])
        self.cache[index] = [instance, table_distribution]
        return instance, table_distribution
