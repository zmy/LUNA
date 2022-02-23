import json, os
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

from .utils import *


class TapasDataset(Dataset):
    def __init__(self, tokenizer, max_seq_length, dataset_dir, use_numtok=1, keep_origin=0):
        assert max_seq_length <= 512
        self.tokenizer = tokenizer
        self.use_numtok = use_numtok
        self.keep_origin = keep_origin
        self.max_seq_length = max_seq_length
        self.tables = {}
        self.texts = {}
        self.pairs = []
        for ddir in dataset_dir:
            with open(os.path.join(ddir, 'all_table.json'), encoding='utf8') as f:
                self.tables.update(json.load(f))
            with open(os.path.join(ddir, 'all_text.json'), encoding='utf8') as f:
                self.texts.update(json.load(f))
            with open(os.path.join(ddir, 'all_pair.json'), encoding='utf8') as f:
                self.pairs.extend(json.load(f) if 'extra' not in ddir else random.sample(json.load(f), 51280))
        self.pairs = np.array(self.pairs, dtype=np.object)
        self.table_numbers = {}
        self.table_distributions = {}
        self.text_numbers = {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        table_id, text_id = self.pairs[index]
        table = self.tables[table_id]
        text = self.texts[text_id]
        number_list = []
        if text_id in self.text_numbers:
            number_list.extend(self.text_numbers[text_id])
        else:
            if self.use_numtok:
                text, _ = parse_string(text, MAX_QUERY_CHARS, MAX_QUERY_NUM, keep_origin=self.keep_origin)
            else:
                _ = []
            self.texts[text_id] = text
            self.text_numbers[text_id] = _
            number_list.extend(_)

        if table_id in self.table_numbers:
            number_list.extend(self.table_numbers[table_id])
            table_distribution = self.table_distributions[table_id]
        else:
            table = table[:MAX_ROW_SIZE]
            table_number_list = []
            col = []
            for j, row in enumerate(table):
                for i in range(len(row)):
                    if self.use_numtok:
                        row[i], _ = parse_string(row[i], MAX_CELL_CHARS, MAX_CELL_NUM, keep_origin=self.keep_origin)
                    else:
                        _ = []
                    table_number_list.extend(_)
                    if j == 0:
                        col.append([])
                    else:
                        if len(_) == 1:
                            col[i].append(_[0][1])
                        else:
                            col[i].append(None)
            table_distribution = []
            for x in col[:MAX_COL_SIZE]:
                table_distribution.append(compute_col_distribution(x))
            while len(table_distribution) < MAX_COL_SIZE:
                table_distribution.append([-1e11] * 5)
            table_distribution = torch.tensor(table_distribution, dtype=torch.float32)
            self.table_distributions[table_id] = table_distribution
            self.tables[table_id] = table
            self.table_numbers[table_id] = table_number_list
            number_list.extend(table_number_list)

        if len(table) == 0:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(table[1:], columns=table[0])

        inputs = self.tokenizer(table=df, queries=text, padding="max_length", return_tensors="pt")
        for k, v in inputs.items():
            v = v.squeeze(0)
            if v.size(0) > self.max_seq_length:
                v = v[:self.max_seq_length]
            inputs[k] = v
        return inputs, number_list[:(inputs.input_ids == self.tokenizer.num_token_id).sum().item()], table_distribution
