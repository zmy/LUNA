import re

import numpy as np
import string
import torch
from typing import List,Dict

# "SPAN-TABLE-TEXT": 2 has no data
OPERATOR_CLASSES_ = {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "CHANGE_RATIO": 3,
                     "AVERAGE": 4, "COUNT": 5, "SUM": 6, "DIFF": 7, "TIMES": 8, "DIVIDE": 9}

id2OPERATOR_CLASSES_={}
for k,v in OPERATOR_CLASSES_.items():
    id2OPERATOR_CLASSES_[v]=k

OPERATOR = ['+', '-', '*', '/']

SCALE = ["", "thousand", "million", "billion", "percent"]
def scale_to_num(scale):
    scale = scale.lower()
    num = 1
    if 'hundred' in scale:  # hundred
        num = 100
    elif 'thousand' in scale:  # thousand
        num = 1000
    elif 'million' in scale:  # million
        num = 1000000
    elif 'billion' in scale:  # billion
        num = 1000000000
    elif 'percent' in scale:  # percent
        num = 0.01
    return num

def extract_one_num_from_str(s):
    s = _clean_num(s)
    r_num = r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)"
    groups = re.findall(r_num, s)
    if len(groups) == 0:
        return None
    num = groups[0][0]
    if num == '':
        return None
    if '.' in num:
        return float(num)
    return int(num)

EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"
def _clean_num(text:str):
    return "".join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])


def is_number(text: str) -> bool:
    try:
        words = " ".join([_clean_num(w) for w in text.split()]).split()
        if len(words) == 0:
            """1023 or 1 million"""
            return False
        num = float(words[0])
        if np.isnan(num):
            return False
        if len(words) >= 2:
            if scale_to_num(words[1]) == 1:
                return False
        return True
    except ValueError:
        return False

def negative_num_handle(x):
    """
    :param x:  transform (134) -> -134
    :return:
    """
    all = re.findall('(\([\d.\s]+\))', x.strip())
    if len(all) > 0:
        return -1
    return 1

def percent_num_handle(x):
    """
    :param x:  transform 12% -> 12/100
    :return:
    """
    all = re.findall('([\d.\s]+%)', x.strip())
    if len(all) > 0:
        return 0.01
    return 1

def word_scale_handle(x):
    """
    :param x: 1 million = 1,000,000
    :return:
    """
    iter = re.finditer('([\d.]+\s?[a-zA-Z]+)', x)
    for one in iter:
        text = one.group(0).lower()
        scale_val = scale_to_num(text)
        return scale_val
    return 1

def to_number(text:str) -> float:
    num = extract_one_num_from_str(text)
    scale_val = word_scale_handle(text)
    negative_flag = negative_num_handle(text)
    percent_flag = percent_num_handle(text)
    if num is not None:
        return round(num * scale_val * negative_flag * percent_flag, 4)
    return None

def remove_articles(text: str) -> str:
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def white_space_fix(text: str) -> str:
    return ' '.join(text.split())

EXCLUDE = set(string.punctuation)
def remove_punc(text: str) -> str:
    if not is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def normalize_number(text: str) -> str:
    if is_number(text):
        return str(to_number(text))
    else:
        return text

def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [white_space_fix(remove_articles(normalize_number(remove_punc(token.lower()))))
             for token in text.split()]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized


class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index
        Args:
            indices (:obj:`torch.LongTensor`, same shape as a `values` Tensor to which the indices refer):
                Tensor containing the indices.
            num_segments (:obj:`torch.LongTensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor must have the same
                number of segments (although many segments can be empty).
            batch_dims (:obj:`int`, `optional`, defaults to 0):
                The number of batch dimensions. The first `batch_dims` dimensions of a SegmentedTensor are treated as
                batch dimensions. Segments in different batch elements are always distinct even if they have the same
                index.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments, device=indices.device)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return self.indices.size()[: self.batch_dims]  # returns a torch.Size object



class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segments` * `inner_index.num_segments`
        Args:
            outer_index (:obj:`IndexMap`):
                IndexMap.
            inner_index (:obj:`IndexMap`):
                IndexMap, must have the same shape as `outer_index`.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super(ProductIndexMap, self).__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=(index.indices // self.inner_index.num_segments).type(torch.float).floor().type(torch.long),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims,
        )

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments)
                .type(torch.float)
                .floor()
                .type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
        )


def convert_start_end_tags(split_tags, paragraph_index):
    in_split_tags = split_tags.copy()
    split_tags = [0 for i in range(len(split_tags))]
    for i in range(len(in_split_tags)):
        if in_split_tags[i] == 1:
            current_index = paragraph_index[i]
            split_tags[i] = 1
            paragraph_index_ = paragraph_index[i:]
            for j in range(1, len(paragraph_index_)):
                if paragraph_index_[j] == current_index:
                    split_tags[i + j] = 1
                else:
                    break
            break
    for i in range(1, len(in_split_tags)):
        if in_split_tags[-i] == 1:
            current_index = paragraph_index[-i]
            split_tags[-i] = 1
            paragraph_index_ = paragraph_index[:-i]
            for j in range(1, len(paragraph_index_)):
                if paragraph_index_[-j] == current_index:
                    split_tags[-i - j] = 1
                else:
                    break
            break
    del in_split_tags
    return split_tags





def sortFunc(elem):
    return elem[1]





def get_answer_nums(table_answer_coordinates: List, paragraph_answer_coordinates: Dict):
    if table_answer_coordinates is not None:
        table_answer_num = len(table_answer_coordinates)
    else:
        table_answer_num = 0
    paragraph_answer_nums = 0
    if paragraph_answer_coordinates:
        for value in paragraph_answer_coordinates.values():
            paragraph_answer_nums += len(value)
    return table_answer_num, paragraph_answer_nums


def get_operands_index(label_ids, token_type_ids):
    row_ids = token_type_ids[:, :, 2]
    column_ids = token_type_ids[:, :, 1]
    max_num_rows = 64
    max_num_columns = 32
    row_index = IndexMap(
        indices=torch.min(row_ids, torch.as_tensor(max_num_rows - 1, device=row_ids.device)),
        num_segments=max_num_rows,
        batch_dims=1,
    )
    col_index = IndexMap(
        indices=torch.min(column_ids, torch.as_tensor(max_num_columns - 1, device=column_ids.device)),
        num_segments=max_num_columns,
        batch_dims=1,
    )
    cell_index = ProductIndexMap(row_index, col_index).indices
    first_operand_start = torch.argmax((label_ids != 0).int(), dim=1)[0]
    label_ids = label_ids[0, first_operand_start:]
    cell_index_first = cell_index[0, first_operand_start:]
    first_operand_end = torch.argmax(((cell_index_first - cell_index[0, first_operand_start]) != 0).int())

    label_ids = label_ids[first_operand_end:]
    cell_index_first = cell_index_first[first_operand_end:]
    first_operand_end = first_operand_end + first_operand_start

    second_operand_start = torch.argmax((label_ids != 0).int())
    cell_index_second = cell_index_first[second_operand_start:]
    second_operand_end = torch.argmax(
        ((cell_index_second - cell_index_first[second_operand_start]) != 0).int()) + second_operand_start
    second_operand_start += first_operand_end
    second_operand_end += first_operand_end
    return first_operand_start, first_operand_end, second_operand_start, second_operand_end


def get_tokens_from_ids(ids, tokenizer):
    tokens = []
    sub_tokens = []
    for id in ids:
        token = tokenizer._convert_id_to_token(id)
        if len(sub_tokens) == 0:
            sub_tokens.append(token)
        elif str(token).startswith("##"):
            sub_tokens.append(token[2:])
        elif len(sub_tokens) != 0:
            tokens.append("".join(sub_tokens))
            sub_tokens = [token]
    tokens.append("".join(sub_tokens))
    return "".join(tokens)


def get_number_mask(table):
    max_num_rows = 64
    max_num_columns = 32
    columns = table.columns.tolist()
    number_mask = np.zeros((1, max_num_columns * max_num_rows))
    number_value = np.ones((1, max_num_columns * max_num_rows)) * np.nan
    for index, row in table.iterrows():
        for col_index in columns:
            col_index = int(col_index)
            in_cell_index = (index + 1) * max_num_columns + col_index + 1
            table_content = row[col_index]
            number = to_number(table_content)
            if number is not None:
                number_mask[0, in_cell_index] = 1
                number_value[0, in_cell_index] = float(number)
    return number_mask, number_value

def get_order_by_tf_idf(question, paragraphs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    sorted_order = []
    corpus = [question]
    for order, text in paragraphs.items():
        corpus.append(text)
        sorted_order.append(order)
    tf_idf = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()[1:]
    sorted_similarities = sorted(enumerate(cosine_similarities), key=lambda x: x[1])
    idx = [i[0] for i in sorted_similarities][::-1]
    return [sorted_order[index] for index in idx]

def get_number_order_labels(paragraphs,paragraph_sorted_order, table, derivation, operator_class, answer_mapping, question_id,
                            OPERATOR_CLASSES,transpose_table=False):
    if ("DIVIDE" not in OPERATOR_CLASSES or operator_class != OPERATOR_CLASSES["DIVIDE"]) and \
            ("CHANGE_RATIO" not in OPERATOR_CLASSES or operator_class != OPERATOR_CLASSES["CHANGE_RATIO"]) and \
            ("DIFF" not in OPERATOR_CLASSES or operator_class != OPERATOR_CLASSES["DIFF"]):
        return -1
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy
    operands = get_operands(derivation)
    first_operand, second_operand = operands[0], operands[1]
    answer_from = answer_mapping.keys()
    table_answer_coordinates = None
    paragraph_answer_coordinates = None
    if "table" in answer_from:
        table_answer_coordinates = answer_mapping["table"]
    if "paragraph" in answer_from:
        paragraph_answer_coordinates = answer_mapping["paragraph"]
    table_answer_nums, paragraph_answer_nums = get_answer_nums(table_answer_coordinates, paragraph_answer_coordinates)
    if (table_answer_nums + paragraph_answer_nums) < 2:
        # print("the same number to skip it: derivation")
        raise RuntimeError(f" skip this the derivation is {derivation} ")
    if table_answer_nums == 2:
        #b2786c1a-37de-4120-b03c-32bf5c81f157   a shit code
        answer_coordinates = answer_mapping["table"]
        answer_coordinates_copy = answer_coordinates.copy()
        answer_coordinates = [(answer_coordinate[0], answer_coordinate[1]) for answer_coordinate in
                              answer_coordinates_copy]
        del answer_coordinates_copy
        operand_one = to_number(table[answer_coordinates[0][0]][answer_coordinates[0][1]])
        operand_two = to_number(table[answer_coordinates[1][0]][answer_coordinates[1][1]])

        if answer_coordinates[0][int(transpose_table)] < answer_coordinates[1][int(transpose_table)]:
            return int(str(operand_one) != str(first_operand))
        elif answer_coordinates[0][int(transpose_table)] == answer_coordinates[1][int(transpose_table)] and \
                answer_coordinates[0][int(not transpose_table)] < answer_coordinates[1][int(not transpose_table)]:
            return int(str(operand_one) != str(first_operand))
        else:
            return int(str(operand_one) == str(first_operand))
    elif paragraph_answer_nums == 2:
        paragraph_mapping_orders = list(answer_mapping["paragraph"].keys())
        if len(paragraph_mapping_orders) == 1:
            answer_one_order, answer_two_order = (paragraph_mapping_orders[0], paragraph_mapping_orders[0])
            answer_one_start = answer_mapping["paragraph"][answer_one_order][0][0]
            answer_one_end = answer_mapping["paragraph"][answer_one_order][0][1]
            answer_two_start = answer_mapping["paragraph"][answer_two_order][1][0]
            answer_two_end = answer_mapping["paragraph"][answer_two_order][1][1]
        else:
            answer_one_order = paragraph_mapping_orders[0]
            answer_two_order = paragraph_mapping_orders[1]
            answer_one_start = answer_mapping["paragraph"][answer_one_order][0][0]
            answer_one_end = answer_mapping["paragraph"][answer_one_order][0][1]
            answer_two_start = answer_mapping["paragraph"][answer_two_order][0][0]
            answer_two_end = answer_mapping["paragraph"][answer_two_order][0][1]
        operand_one = to_number(paragraphs[int(answer_one_order)][answer_one_start:answer_one_end])
        operand_two = to_number(paragraphs[int(answer_two_order)][answer_two_start:answer_two_end])
        if paragraph_sorted_order.index(int(answer_one_order)) < paragraph_sorted_order.index(int(answer_two_order)):
            return int(operand_one != first_operand)
        elif paragraph_sorted_order.index(int(answer_one_order)) == paragraph_sorted_order.index(int(answer_two_order)) and answer_one_start < answer_two_start:
            return int(operand_one != first_operand)
        else:
            return int(operand_one == first_operand)
    else:
        answer_coordinates = answer_mapping["table"]
        operand_one = to_number(table[answer_coordinates[0][0]][answer_coordinates[0][1]])
        paragraph_mapping_orders = list(answer_mapping["paragraph"].keys())
        answer_two_order = paragraph_mapping_orders[0]
        answer_two_start = answer_mapping["paragraph"][answer_two_order][0][0]
        answer_two_end = answer_mapping["paragraph"][answer_two_order][0][1]
        operand_two = to_number(paragraphs[int(answer_two_order)][answer_two_start:answer_two_end])
        return int(operand_one != first_operand)

def get_operators(derivation: str):
    res = []
    for c in derivation:
        if c in OPERATOR:
            res.append(c)
    return res


def get_operands(derivation):
    num_strs = re.split('\+|-|\*|/', derivation)
    result = []
    for it in num_strs:
        one = to_number(it)
        if one is not None:
            result.append(one)
    return result


def facts_to_nums(facts):
    return [to_number(f) for f in facts]

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _is_average(num_facts: list, answer):
    return round(np.average(num_facts), 2) == round(answer, 2)


def _is_change_ratio(num_facts: list, answer):
    if len(num_facts) != 2:
        return False
    cands = []
    if num_facts[1] != 0:
        ori_percent = round(100 * (num_facts[0] - num_facts[1]) / num_facts[1], 2)
        cands.append(ori_percent)
    if num_facts[0] != 0:
        ori_percent = round(100 * (num_facts[1] - num_facts[0]) / num_facts[0], 2)
        cands.append(ori_percent)
    return round(answer, 2) in cands


def _is_division(num_facts: list, answer):
    if len(num_facts) != 2:
        return False
    cands = []
    if num_facts[1] != 0:
        cands.append(round(num_facts[0] / num_facts[1], 2))
        cands.append(100 * round(num_facts[0] / num_facts[1], 2))
    if num_facts[0] != 0:
        cands.append(round(num_facts[1] / num_facts[0], 2))
        cands.append(100 * round(num_facts[1] / num_facts[0], 2))
    return round(answer, 2) in cands


def _is_diff(num_facts: list, answer):
    if len(num_facts) != 2:
        return False
    ans_1 = round(num_facts[0] - num_facts[1], 2)
    ans_2 = round(num_facts[1] - num_facts[0], 2)
    return round(answer, 2) in (ans_1, ans_2)


def _is_sum(num_facts: list, answer):
    return round(np.sum(num_facts), 2) == round(answer, 2)


def _is_times(num_facts: list, answer):
    return round(np.prod(num_facts), 2) == round(answer, 2)


def get_operator_class(derivation: str, answer_type: str, facts: list, answer, mapping: dict, scale, OPERATOR_CLASSES):
    operator_class = None
    try:
        if answer_type == "span":
            if "table" in mapping:
                operator_class = OPERATOR_CLASSES["SPAN-TABLE"]
            else:
                operator_class = OPERATOR_CLASSES["SPAN-TEXT"]
        elif answer_type == "multi-span":
            operator_class = OPERATOR_CLASSES["MULTI_SPAN"]
        elif answer_type == "count":
            operator_class = OPERATOR_CLASSES["COUNT"]
        elif answer_type == "arithmetic":
            num_facts = facts_to_nums(facts)
            if not is_number(str(answer)):
                return None  # not support date
            if _is_change_ratio(num_facts, answer):
                operator_class = OPERATOR_CLASSES["CHANGE_RATIO"]
            elif _is_average(num_facts, answer):
                operator_class = OPERATOR_CLASSES["AVERAGE"]
            elif _is_sum(num_facts, answer):
                operator_class = OPERATOR_CLASSES["SUM"]
            elif _is_times(num_facts, answer):
                operator_class = OPERATOR_CLASSES["TIMES"]
            elif _is_diff(num_facts, answer):
                operator_class = OPERATOR_CLASSES["DIFF"]
            elif _is_division(num_facts, answer):
                operator_class = OPERATOR_CLASSES["DIVIDE"]

            operators = get_operators(derivation)
            if len(operators) == 1:  # if it is detected that only have one operator, use the one in the derivation
                if operators[0] == "/":
                    return OPERATOR_CLASSES["DIVIDE"]
                elif operators[0] == "-":
                    operator_class = OPERATOR_CLASSES["DIFF"]
                elif operators[0] == "*":
                    operator_class = OPERATOR_CLASSES["TIMES"]
                elif operators[0] == "+":
                    operator_class = OPERATOR_CLASSES["SUM"]
    except KeyError:
        operator_class = None
    return operator_class
