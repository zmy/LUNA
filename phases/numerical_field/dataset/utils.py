import math

from number_tokenizer.numtok import NumTok

MAX_QUERY_CHARS = 256
MAX_QUERY_NUM = 7
MAX_CELL_CHARS = 128
MAX_CELL_NUM = 3
MAX_ROW_SIZE = 20
MAX_COL_SIZE = 50


def parse_string(text, max_seq_length=500, max_num=20, num_token='[NUM]', do_lower_case=True, keep_origin=0):
    text = str(text)
    if do_lower_case:
        text = text.lower()
        num_token = num_token.lower()
    while num_token in text:
        text = text.replace(num_token, '')
    text = text[:max_seq_length]
    vals = sorted(NumTok.find_numbers(text), key=lambda x: x[1])
    new_text = ""
    number_list = []
    start_point = 0
    for val in vals:
        if val[1] >= start_point:
            num = NumTok.get_val(val[0])
            if math.isnan(num):
                continue
            number_list.append((val[0], num))
            new_text += text[start_point:val[1]] + (val[0] if keep_origin else '') + num_token
            start_point = val[2]
            if len(number_list) == max_num:
                break
    new_text += text[start_point:]
    return new_text, number_list


def compute_col_distribution(x, ret_sum=False):
    _x = []
    for i in x:
        if i is not None:
            _x.append(i)
    # min,max,avg,01range,0100range
    if len(_x) / (len(x) + 1e-7) > 0.5:
        minn = min(_x)
        maxx = max(_x)
        avg = sum(_x) / len(_x)
        rg_01 = sum([1 for i in _x if math.fabs(i) <= 1]) / (len(x) + 1e-7)
        rg_0100 = sum([1 for i in _x if math.fabs(i) <= 100]) / (len(x) + 1e-7)
        ret = [minn, maxx, avg, rg_01, rg_0100]
        if ret_sum:
            ret.append(sum(_x))
    else:
        ret = [-1e11] * 5
        if ret_sum:
            ret.append(-1e11)
    return ret
