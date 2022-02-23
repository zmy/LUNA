import json, os
import math

import numpy as np
import torch
from torch.utils.data import Dataset
import random

from .utils import *


class RobertaDataset(Dataset):
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
        input_ids = []
        column_ids = []
        if text_id in self.text_numbers:
            number_list.extend(self.text_numbers[text_id])
        else:
            if self.use_numtok:
                text, _ = parse_string(text, MAX_QUERY_CHARS, MAX_QUERY_NUM, do_lower_case=False,
                                       keep_origin=self.keep_origin)  # query should not be longer than 256 chars, and at most 7 numbers in it
            else:
                _ = []
            text = self.tokenizer.encode(text)
            self.texts[text_id] = text
            self.text_numbers[text_id] = _
            number_list.extend(_)
        input_ids.extend(text)
        column_ids.extend([0] * len(text))

        if table_id in self.table_numbers:
            number_list.extend(self.table_numbers[table_id])
            table_distribution = self.table_distributions[table_id]
        else:
            table = table[:MAX_ROW_SIZE]  # max row size 20
            column_table = []
            for j, row in enumerate(table):
                for i in range(len(row)):
                    if j == 0:
                        column_table.append([])
                    column_table[i].append(row[i])

            table_ids = []
            table_column_ids = []
            table_number_list = []
            column_values = []
            for i, col in enumerate(column_table):
                column_values.append([])
                for j in range(len(col)):
                    if self.use_numtok:
                        cell_text, _ = parse_string(col[j], MAX_CELL_CHARS, MAX_CELL_NUM, do_lower_case=False,
                                                    keep_origin=self.keep_origin)  # cell should not be longer than 128 chars, and at most 3 numbers in it
                    else:
                        cell_text, _ = col[j], []
                    cell_text = self.tokenizer.encode(cell_text)[1:-1]
                    table_number_list.extend(_)
                    table_ids.extend(cell_text)
                    table_column_ids.extend([i + 1] * len(cell_text))
                    if len(_) == 1:
                        column_values[i].append(_[0][1])
                    else:
                        column_values[i].append(None)

            table_distribution = []
            for x in column_values[:MAX_COL_SIZE]:  # max column size 50 to compute distribution(deprecated)
                table_distribution.append(compute_col_distribution(x))
            while len(table_distribution) < MAX_COL_SIZE:
                table_distribution.append([-1e11] * 5)
            table_distribution = torch.tensor(table_distribution, dtype=torch.float32)
            self.table_distributions[table_id] = table_distribution
            table_ids.append(self.tokenizer.sep_token_id)
            table_column_ids.append(0)
            table = (table_ids, table_column_ids)
            self.tables[table_id] = table
            self.table_numbers[table_id] = table_number_list
            number_list.extend(table_number_list)
        input_ids.extend(table[0])
        column_ids.extend(table[1])

        if len(input_ids) > self.max_seq_length:
            mask = [1] * self.max_seq_length
            input_ids = input_ids[:self.max_seq_length]
            column_ids = column_ids[:self.max_seq_length]
        else:
            mask = [1] * len(input_ids) + [0] * (self.max_seq_length - len(input_ids))
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
            column_ids = column_ids + [0] * (self.max_seq_length - len(column_ids))
        mask = torch.tensor(mask, dtype=torch.int64)
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        column_ids = torch.tensor(column_ids, dtype=torch.int64)

        return {'input_ids': input_ids, 'attention_mask': mask, 'column_ids': column_ids}, number_list[:(
                input_ids == self.tokenizer.num_token_id).sum().item()], table_distribution
