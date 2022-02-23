from typing import Dict, List

import numpy as np
import torch

from number_tokenizer import NumTok
from .data_util import get_number_order_labels, to_number, is_number, \
    get_operator_class, SCALE
from .roberta_dataset import RobertaReader, string_tokenizer, paragraph_tokenize, question_tokenizer, truncation


def table_tokenize(table, tokenizer, mapping, use_numtok):
    table_cell_tokens = []
    table_ids = []
    table_tags = []
    table_cell_index = []
    table_number_strings = []
    table_cell_number_value = []
    table_mapping = False
    answer_coordinates = None

    if "table" in mapping and len(mapping["table"]) != 0:
        table_mapping = True
        answer_coordinates = mapping["table"]
    column_ids = []
    row_ids = []
    current_cell_index = 1
    if len(table) > 0:
        for j in range(len(table[0])):
            for i in range(len(table)):
                cell_ids, number_strings = string_tokenizer(table[i][j], tokenizer, use_numtok=use_numtok,
                                                            do_lower_case=True)
                if not cell_ids:
                    continue
                table_ids += cell_ids
                if is_number(table[i][j]):
                    table_cell_number_value.append(to_number(table[i][j]))
                else:
                    table_cell_number_value.append(np.nan)
                table_cell_tokens.append(table[i][j])
                if table_mapping and [i, j] in answer_coordinates:
                    table_tags += [1] * len(cell_ids)
                else:
                    table_tags += [0] * len(cell_ids)
                table_cell_index += [current_cell_index] * len(cell_ids)
                column_ids += [i + 1] * len(cell_ids)
                row_ids += [j] * len(cell_ids)
                current_cell_index += 1
                table_number_strings += number_strings  # NEWLY ADDED

    return table_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index, table_number_strings, column_ids, row_ids


def _concat(question_ids,
            question_number_strings,
            table_ids,
            column_ids,
            row_ids,
            table_tags,
            table_cell_index,
            table_number_strings,
            paragraph_ids,
            paragraph_tags,
            paragraph_index,
            paragraph_number_strings,
            tokenizer,
            sep_start,
            sep_end,
            question_length_limitation,
            passage_length_limitation):
    if question_length_limitation is not None:
        if len(question_ids) > question_length_limitation:
            question_number_strings, question_ids = truncation(tokenizer, question_length_limitation,
                                                               question_number_strings, question_ids)
    if passage_length_limitation is not None:
        if len(table_ids) > passage_length_limitation:
            table_number_strings, table_ids, column_ids, row_ids, table_tags, table_cell_index = \
                truncation(tokenizer, passage_length_limitation, table_number_strings, table_ids, column_ids, row_ids,
                           table_tags, table_cell_index)
        if len(paragraph_ids) > passage_length_limitation - len(table_ids):
            paragraph_number_strings, paragraph_ids, paragraph_tags, paragraph_index = \
                truncation(tokenizer, passage_length_limitation - len(table_ids), paragraph_number_strings,
                           paragraph_ids, paragraph_tags, paragraph_index)

    input_ids = [sep_start] + question_ids + [sep_end] + table_ids + [sep_end] + paragraph_ids + [sep_end]
    input_segments = [0] * (len(question_ids) + 2) + [1] * (len(table_ids) + 1) + [2] * (len(paragraph_ids) + 1)
    input_index = [0] * (len(question_ids) + 2) + table_cell_index + [0] + paragraph_index + [0]
    tags = [0] * (len(question_ids) + 2) + table_tags + [0] + paragraph_tags + [0]

    input_segments_for_encoder = [0] * (len(question_ids) + 2) + [1] * (len(table_ids) + 1) + [0] * (
            len(paragraph_ids) + 1)
    column_ids = [0] * (len(question_ids) + 2) + column_ids + [0] + [0] * (len(paragraph_ids) + 1)
    row_ids = [0] * (len(question_ids) + 2) + row_ids + [0] + [0] * (len(paragraph_ids) + 1)
    token_type_ids_for_encoder = [input_segments_for_encoder, column_ids, row_ids]

    number_strings = question_number_strings + table_number_strings + paragraph_number_strings
    return input_ids, number_strings, input_segments, input_index, tags, token_type_ids_for_encoder


def collate(data, tokenizer, kept_keys, encoder):
    bsz = len(data)
    if encoder == "bert":
        for x in data:
            if len(x["input_ids"]) == 513:
                for key in ["input_ids", "token_type_ids", "input_index", "tag_labels"]:
                    x[key].pop(-1)
                for _x in x['token_type_ids_for_encoder']:
                    _x.pop(-1)

    max_length = max([len(x["input_ids"]) for x in data])
    input_ids = torch.LongTensor(bsz, max_length).fill_(tokenizer.pad_token_id)
    number_strings = []
    attention_mask = torch.zeros(bsz, max_length, dtype=torch.int64)
    token_type_ids = torch.zeros(bsz, max_length, dtype=torch.int64)
    token_type_ids_for_encoder = torch.zeros(bsz, max_length, 7, dtype=torch.int64)
    input_index = torch.zeros(bsz, max_length, dtype=torch.int64)
    tag_labels = torch.zeros(bsz, max_length, dtype=torch.int64)
    operator_labels = torch.LongTensor(bsz)
    scale_labels = torch.LongTensor(bsz)
    number_order_labels = torch.LongTensor(bsz)
    paragraph_tokens = []
    table_cell_tokens = []
    gold_answers = []
    paragraph_numbers = []
    table_cell_numbers = []
    question_ids = []
    for i, _data in enumerate(data):
        input_ids[i, :len(_data['input_ids'])] = torch.tensor(_data['input_ids'])
        attention_mask[i, :len(_data['input_ids'])] = 1
        token_type_ids[i, :len(_data['token_type_ids'])] = torch.tensor(_data['token_type_ids'])
        input_index[i, :len(_data['input_index'])] = torch.tensor(_data['input_index'])
        tag_labels[i, :len(_data['tag_labels'])] = torch.tensor(_data['tag_labels'])
        token_type_ids_for_encoder[i, :len(_data['input_ids']), :3] = torch.tensor(
            _data['token_type_ids_for_encoder']).T
        operator_labels[i] = _data['operator_label']
        number_order_labels[i] = _data['number_order_label']
        scale_labels[i] = _data['scale_label']
        number_strings.extend(_data['number_strings'])
        table_cell_tokens.append(_data['table_cell_tokens'])
        paragraph_tokens.append(_data['paragraph_tokens'])
        table_cell_numbers.append(_data['table_cell_number_value'])
        paragraph_numbers.append(_data['paragraph_number_value'])
        gold_answers.append(_data['answer_dict'])
        question_ids.append(_data['question_id'])
    if encoder == 'bert':
        token_type_ids_for_encoder = token_type_ids_for_encoder[..., 0]
    out_batch = {
        "input_ids": input_ids, "numtok_dict": NumTok.collate(number_strings, kept_keys), 'input_index': input_index,
        "attention_mask": attention_mask, "token_type_ids": token_type_ids,
        "tag_labels": tag_labels, "operator_labels": operator_labels, "scale_labels": scale_labels,
        "number_order_labels": number_order_labels,
        "paragraph_tokens": paragraph_tokens, "table_cell_tokens": table_cell_tokens,
        "token_type_ids_for_encoder": token_type_ids_for_encoder,
        "paragraph_numbers": paragraph_numbers,
        "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers,
        "question_ids": question_ids,
    }

    return out_batch


class TapasReader(RobertaReader):
    def _to_instance(self, question: str, table: List[List[str]], paragraphs: List[Dict], answer_from: str,
                     answer_type: str, answer: str, derivation: str, facts: list, answer_mapping: Dict, scale: str,
                     question_id: str, use_numtok: int):  # CHANGED
        operator_class = get_operator_class(derivation, answer_type, facts, answer,
                                            answer_mapping, scale, self.OPERATOR_CLASSES)
        scale_class = SCALE.index(scale)

        if operator_class is None:
            return None

        table_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index, table_number_strings, column_ids, row_ids = \
            table_tokenize(table, self.tokenizer, answer_mapping, use_numtok)

        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                    table[i][j] = "NONE"

        paragraph_tokens, paragraph_ids, paragraph_tags, paragraph_number_value, paragraph_index, paragraph_number_strings, paragraph_sorted_order = \
            paragraph_tokenize(question, paragraphs, self.tokenizer, answer_mapping, use_numtok, do_lower_case=True)

        question_ids, question_num_strings = question_tokenizer(question, self.tokenizer, use_numtok,
                                                                do_lower_case=True)

        number_order_label = get_number_order_labels(paragraphs, paragraph_sorted_order, table, derivation,
                                                     operator_class,
                                                     answer_mapping, question_id, self.OPERATOR_CLASSES,
                                                     transpose_table=True)

        input_ids, number_strings, token_type_ids, input_index, tags, token_type_ids_for_encoder = \
            _concat(question_ids, question_num_strings, table_ids, column_ids, row_ids, table_tags,
                    table_cell_index, table_number_strings,
                    paragraph_ids, paragraph_tags, paragraph_index, paragraph_number_strings,
                    self.tokenizer,
                    self.sep_start, self.sep_end, self.question_length_limit,
                    self.passage_length_limit)
        answer_dict = {"answer_type": answer_type, "answer": answer, "scale": scale, "answer_from": answer_from}
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "input_index": input_index,
            "paragraph_number_value": paragraph_number_value,
            "table_cell_number_value": table_cell_number_value,
            "tag_labels": tags,
            "number_order_label": int(number_order_label),
            "operator_label": int(operator_class),
            "scale_label": int(scale_class),
            "paragraph_tokens": paragraph_tokens,
            "table_cell_tokens": table_cell_tokens,
            "token_type_ids_for_encoder": token_type_ids_for_encoder,
            "answer_dict": answer_dict,
            "question_id": question_id,
            "number_strings": number_strings
        }
