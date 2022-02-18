import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from number_tokenizer import NumTok
from .data_util import _is_whitespace, get_order_by_tf_idf, get_number_order_labels, to_number, is_number, \
    OPERATOR_CLASSES_, get_operator_class, SCALE


def string_tokenizer(string: str, tokenizer: PreTrainedTokenizer, use_numtok: int, do_lower_case: bool) -> Tuple[
    List[int], List[str]]:
    if not string:
        return [], []
    if use_numtok:
        new_text, number_triplets = NumTok.replace_numbers(string, do_lower_case=do_lower_case,
                                                           keep_origin=int(use_numtok == 2))  # NEWLY ADDED SECTION
        number_strings = [t[0] for t in number_triplets]
    else:
        new_text, number_strings = string, []

    tokens = tokenizer.tokenize(new_text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids, number_strings


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

    current_cell_index = 1
    for i in range(len(table)):
        for j in range(len(table[i])):
            cell_ids, number_strings = string_tokenizer(table[i][j], tokenizer, use_numtok=use_numtok,
                                                        do_lower_case=False)
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
            current_cell_index += 1
            table_number_strings += number_strings  # NEWLY ADDED

    return table_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index, table_number_strings


def paragraph_tokenize(question, paragraphs, tokenizer, mapping, use_numtok, do_lower_case=False):
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy
    split_tags = []
    number_values = []
    number_strings = []
    tokens = []
    tags = []
    paragraph_index = []

    paragraph_mapping = False
    paragraph_mapping_orders = []
    if "paragraph" in mapping and len(mapping["paragraph"]) != 0:
        paragraph_mapping = True
        paragraph_mapping_orders = list(mapping["paragraph"].keys())
    # apply tf-idf to calculate text-similarity
    sorted_order = get_order_by_tf_idf(question, paragraphs)
    for order in sorted_order:
        text = paragraphs[order]
        prev_is_whitespace = True
        answer_indexs = None
        if paragraph_mapping and str(order) in paragraph_mapping_orders:
            answer_indexs = mapping["paragraph"][str(order)]
        current_tags = [0] * len(text)
        if answer_indexs is not None:
            for answer_index in answer_indexs:
                current_tags[answer_index[0]:answer_index[1]] = \
                    [1] * len(current_tags[answer_index[0]:answer_index[1]])

        start_index = 0
        wait_add = False
        for i, c in enumerate(text):
            if _is_whitespace(c):  # or c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                prev_is_whitespace = True
            elif c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                tokens.append(c)
                tags.append(0)
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
        if wait_add:
            if 1 in current_tags[start_index:len(text)]:
                tags.append(1)
            else:
                tags.append(0)

    try:
        assert len(tokens) == len(tags)
    except AssertionError:
        print(len(tokens), len(tags))
        input()

    current_token_index = 1
    paragraph_ids = []
    for i, token in enumerate(tokens):
        if use_numtok:
            new_token, number_triplet = NumTok.replace_numbers(token, "[NUM]", do_lower_case=do_lower_case,
                                                               keep_origin=int(use_numtok == 2))
        else:
            new_token, number_triplet = token, []

        for number in number_triplet:
            number_strings.append(number[0])

        number_value = to_number(token)
        if number_value is not None:
            number_values.append(float(number_value))
        else:
            number_values.append(np.nan)

        token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_token if i == 0 else " " + new_token))
        for token_id in token_ids:
            split_tags.append(tags[i])
            paragraph_ids.append(token_id)
            paragraph_index.append(current_token_index)
        current_token_index += 1
    return tokens, paragraph_ids, split_tags, number_values, paragraph_index, number_strings, sorted_order


def question_tokenizer(question_text, tokenizer, use_numtok, do_lower_case=False):
    return string_tokenizer(question_text, tokenizer, use_numtok=use_numtok, do_lower_case=do_lower_case)


def truncation(tokenizer, length, number_strings, ids, *args):
    ids = ids[:length]
    args = list(args)
    for i in range(len(args)):
        args[i] = args[i][:length]
    number_strings = number_strings[:sum([1 for id in ids if id == tokenizer.num_token_id])]
    return (number_strings, ids, *args)


def _concat(question_ids,
            question_number_strings,
            table_ids,
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
            table_number_strings, table_ids, table_tags, table_cell_index = \
                truncation(tokenizer, passage_length_limitation, table_number_strings, table_ids, table_tags,
                           table_cell_index)
        if len(paragraph_ids) > passage_length_limitation - len(table_ids):
            paragraph_number_strings, paragraph_ids, paragraph_tags, paragraph_index = \
                truncation(tokenizer, passage_length_limitation - len(table_ids), paragraph_number_strings,
                           paragraph_ids, paragraph_tags, paragraph_index)

    input_ids = [sep_start] + question_ids + [sep_end] + table_ids + [sep_end] + paragraph_ids + [sep_end]
    input_segments = [0] * (len(question_ids) + 2) + [1] * (len(table_ids) + 1) + [2] * (len(paragraph_ids) + 1)
    input_index = [0] * (len(question_ids) + 2) + table_cell_index + [0] + paragraph_index + [0]
    tags = [0] * (len(question_ids) + 2) + table_tags + [0] + paragraph_tags + [0]
    number_strings = question_number_strings + table_number_strings + paragraph_number_strings
    return input_ids, number_strings, input_segments, input_index, tags


def collate(data, tokenizer, kept_keys):
    bsz = len(data)
    max_length = max([len(x["input_ids"]) for x in data])
    input_ids = torch.LongTensor(bsz, max_length).fill_(tokenizer.pad_token_id)
    number_strings = []
    attention_mask = torch.zeros(bsz, max_length, dtype=torch.int64)
    token_type_ids = torch.zeros(bsz, max_length, dtype=torch.int64)
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
    out_batch = {
        "input_ids": input_ids, "numtok_dict": NumTok.collate(number_strings, kept_keys), 'input_index': input_index,
        "attention_mask": attention_mask, "token_type_ids": token_type_ids,
        "tag_labels": tag_labels, "operator_labels": operator_labels, "scale_labels": scale_labels,
        "number_order_labels": number_order_labels,
        "paragraph_tokens": paragraph_tokens, "table_cell_tokens": table_cell_tokens,
        "paragraph_numbers": paragraph_numbers,
        "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers,
        "question_ids": question_ids,
    }

    return out_batch


class RobertaReader:
    def __init__(self, tokenizer, passage_length_limit: int = None, question_length_limit: int = None,
                 sep_start="<s>", sep_end="<s>"):
        self.tokenizer = tokenizer
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep_start = self.tokenizer._convert_token_to_id(sep_start)
        self.sep_end = self.tokenizer._convert_token_to_id(sep_end)
        self.OPERATOR_CLASSES = OPERATOR_CLASSES_

    def _read(self, file_path: str, use_numtok: int):
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        instances = []
        key_error_count = 0
        index_error_count = 0
        assert_error_count = 0
        skip_count = 0
        for one in tqdm(dataset):
            table = one['table']['table']
            paragraphs = one['paragraphs']
            questions = one['questions']

            for question_answer in questions:
                try:
                    question = question_answer["question"].strip()
                    answer_type = question_answer["answer_type"]
                    derivation = question_answer["derivation"]
                    answer = question_answer["answer"]
                    answer_mapping = question_answer["mapping"]
                    facts = question_answer["facts"]
                    answer_from = question_answer["answer_from"]
                    scale = question_answer["scale"]
                    instance = self._to_instance(question, table, paragraphs, answer_from,
                                                 answer_type, answer, derivation, facts, answer_mapping, scale,
                                                 question_answer["uid"], use_numtok)
                    if instance is not None:
                        instances.append(instance)
                    else:
                        skip_count += 1
                        print("SkipError. Total Error Count: {}".format(skip_count))
                except RuntimeError as e:
                    print(f"run time error:{e}")
                except KeyError:
                    key_error_count += 1
                    print("KeyError. Total Error Count: {}".format(key_error_count))
                except IndexError:
                    index_error_count += 1
                    print("IndexError. Total Error Count: {}".format(index_error_count))
                except AssertionError:
                    assert_error_count += 1
                    print("AssertError. Total Error Count: {}".format(assert_error_count))

        return instances

    def _to_instance(self, question: str, table: List[List[str]], paragraphs: List[Dict], answer_from: str,
                     answer_type: str, answer: str, derivation: str, facts: list, answer_mapping: Dict, scale: str,
                     question_id: str, use_numtok: int):  # CHANGED
        operator_class = get_operator_class(derivation, answer_type, facts, answer,
                                            answer_mapping, scale, self.OPERATOR_CLASSES)
        scale_class = SCALE.index(scale)

        if operator_class is None:
            return None

        table_cell_tokens, table_ids, table_tags, table_cell_number_value, table_cell_index, table_number_strings = \
            table_tokenize(table, self.tokenizer, answer_mapping, use_numtok)

        for i in range(len(table)):
            for j in range(len(table[i])):
                if table[i][j] == '' or table[i][j] == 'N/A' or table[i][j] == 'n/a':
                    table[i][j] = "NONE"

        paragraph_tokens, paragraph_ids, paragraph_tags, paragraph_number_value, paragraph_index, paragraph_number_strings, paragraph_sorted_order = \
            paragraph_tokenize(question, paragraphs, self.tokenizer, answer_mapping, use_numtok)

        question_ids, question_num_strings = question_tokenizer(question, self.tokenizer, use_numtok)

        number_order_label = get_number_order_labels(paragraphs, paragraph_sorted_order, table, derivation,
                                                     operator_class,
                                                     answer_mapping, question_id, self.OPERATOR_CLASSES)

        input_ids, number_strings, token_type_ids, input_index, tags = \
            _concat(question_ids, question_num_strings, table_ids, table_tags,
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
            "answer_dict": answer_dict,
            "question_id": question_id,
            "number_strings": number_strings
        }
