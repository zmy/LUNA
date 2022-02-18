import math
import os.path
import re
from decimal import Decimal
from typing import List, Tuple, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import TapasTokenizer, RobertaTokenizer, BertTokenizer, PreTrainedTokenizer

from .vocab import NumVocab, SPECIAL_NUM_CHARS, DIGIT_CHARS


class NumTok:
    SEGMENT_CHARS = "0123456789+-,.%"
    SEP_SINGLE = ",-.+"
    SEP_CHARS = [",", "-", ".", "+", "-+"]

    @classmethod
    def collate(cls, number_strings: List[str],
                kept_keys: Tuple[str] = ('batch_token_ids', 'batch_seq_len'), device='cpu', fp16=False) -> Dict:
        """
        :param number_strings: A batch of number strings.
        :param kept_keys: The dict keys to kept. Comes from NumBed.param_keys.
        :param device: Tensor device.
        :return: Input dict to NumBed.
        """
        numtok_dict = cls.tokenize([(x, -1, -1) for x in number_strings], device, kept_keys, fp16)
        return {key: numtok_dict[key] for key in kept_keys if key in numtok_dict}

    @classmethod
    def tokenize(cls,
                 numbers_and_indices: List[Tuple[str, int, int]],
                 device='cpu',
                 kept_keys=(),
                 fp16=False) -> Dict:
        if len(numbers_and_indices) == 0:
            return {}
        numbers = [_[0] for _ in numbers_and_indices]
        batch_token_ids = []
        batch_seq_len = []
        batch_sig = []
        batch_exp = []
        batch_length = []
        batch_int = []
        batch_negative = []
        batch_percentage = []
        ret = {}

        if 'batch_digit_mapping' in kept_keys:
            batch_digit_mapping = unpack_and_pad_number_property(get_number_digit_mapping_from_number_list(numbers),
                                                                 device=device)
            ret['batch_digit_mapping'] = batch_digit_mapping

        if 'batch_tuta_feat' in kept_keys:
            batch_tuta_feat = [_.split('|') for _ in get_tuta_feat(numbers)]
            batch_tuta_feat = torch.tensor(list(map(lambda x: list(map(int, x)), batch_tuta_feat))).to(device)
            ret['batch_tuta_feat'] = batch_tuta_feat

        for number in numbers:
            sig, exp = cls.get_sci(number)
            token_ids = torch.tensor([NumVocab.char2idx(c) for c in number]).long()
            batch_token_ids.append(token_ids)
            batch_seq_len.append(len(token_ids))
            batch_sig.append(sig)
            batch_exp.append(exp)

        batch_seq_len = torch.tensor(batch_seq_len).to(device)
        batch_token_ids = pad_sequence(batch_token_ids, batch_first=True, padding_value=NumVocab.PAD).to(device)
        dtype = torch.float32 if not fp16 else torch.float16
        batch_sig = torch.tensor(batch_sig, dtype=dtype).to(device)
        batch_exp = torch.tensor(batch_exp).to(device)

        batch_length, batch_int, batch_negative, batch_percentage = get_format_feat(numbers)
        batch_length = torch.tensor(batch_length).to(device)
        batch_int = torch.tensor(batch_int).to(device)
        batch_negative = torch.tensor(batch_negative).to(device)
        batch_percentage = torch.tensor(batch_percentage).to(device)
        batch_format_feat = (batch_length, batch_int, batch_negative, batch_percentage)
        ret.update({
            'batch_token_ids': batch_token_ids,
            'batch_seq_len': batch_seq_len,
            'batch_sig': batch_sig,
            'batch_exp': batch_exp,
            'batch_format_feat': batch_format_feat
        })
        return ret

    @classmethod
    def find_numbers(cls, string: str) -> List[Tuple[str, int, int]]:
        # print("Segments:", cls._get_segments(string))
        return cls._get_numbers(cls._get_segments(string))

    @classmethod
    def replace_numbers(cls, string: str,
                        num_replace_str: str = "[NUM]",
                        max_seq_length: int = 10000,
                        max_num: int = 10000,
                        do_lower_case: bool = False,
                        keep_origin: int = 0) -> Tuple[str, List[Tuple[str, int, int]]]:
        """
        :param max_seq_length: max number of char in input string
        :param max_num: max number of number in input string
        :param do_lower_case: whether to lowercase the string and replace_str
        :param keep_origin: whether to use exNum
        :return: [<number_string, start_pos, end_pos>].
        """
        if not string:
            return string, []
        string = str(string)
        if do_lower_case:
            string = string.lower()
            num_replace_str = num_replace_str.lower()
        while num_replace_str in string:
            string = string.replace(num_replace_str, "")
        string = string[:max_seq_length]
        numbers_and_indexes = sorted(NumTok.find_numbers(string), key=lambda x: x[1])
        number_list = []
        substituted = ""
        start_point = 0
        for number in numbers_and_indexes:
            if number[1] >= start_point:
                num = NumTok.get_val(number[0])
                if math.isnan(num):
                    continue
                substituted += string[start_point:number[1]] + (number[0] if keep_origin else '') + num_replace_str
                start_point = number[2]
                number_list.append(number)
                if len(number_list) == max_num:
                    break
        substituted += string[start_point:]

        return substituted, number_list

    @classmethod
    def get_val(cls, num_str: str) -> float:
        return float(cls.get_dec(num_str))

    @classmethod
    def get_sci(cls, num_str: str) -> Tuple[float, int]:
        """
        :param num_str: number string
        :return: significand (coefficient) and exponent
        """
        num_dec = cls.get_dec(num_str)
        sig, exp = '{:e}'.format(num_dec).split('e')
        return float(sig), int(exp)

    @staticmethod
    def get_dec(num_str: str) -> Decimal:
        if num_str[-1] == '%':
            num_str = num_str[:-1]
            percent = True
        else:
            percent = False
        num_str = num_str.replace(',', '')
        num_dec = Decimal(num_str) / Decimal(100) \
            if percent else Decimal(num_str)
        return num_dec

    @classmethod
    def _get_segments(cls, text: str) -> List[Tuple[str, int, bool]]:
        """
        Return legal segments in the text.
        :param text:
        :return:
        """
        all_segments = []
        segment = []
        segment_start = 0

        def try_add_segment(seg: List):
            has_digits = not all(c in SPECIAL_NUM_CHARS for c in seg)  # Exclude trivial cases
            if len(seg) > 0 and has_digits:
                prev_space = (segment_start == 0 or text[segment_start - 1].isspace())
                seg = (''.join(seg), segment_start, prev_space)
                all_segments.append(seg)

        for char_idx, char in enumerate(text):
            if len(segment) == 0:
                segment_start = char_idx
            if char in cls.SEGMENT_CHARS and char != '':
                segment.append(char)
            else:
                try_add_segment(segment)
                segment = []
        try_add_segment(segment)

        return all_segments

    @classmethod
    def _get_numbers(cls, segments: List[Tuple[str, int, bool]]) -> List[Tuple[str, int, int]]:
        numbers = []
        segments = [_ for _ in segments if len(_[0]) > 0]
        for seg, idx, prev in segments:
            if not prev and seg[0] in cls.SEP_SINGLE:  # the segment start with a possible separator
                seg_new = seg[1:]
                idx_new = idx + 1
            else:
                seg_new = seg
                idx_new = idx
            if cls._is_legal(seg_new, 'simple'):
                numbers.append((seg_new, idx_new, idx + len(seg)))
            else:
                sep = cls._is_comb(seg)
                if sep:
                    parts = cls._split_seps(seg, idx, sep)
                    s = [(part, idx, (prev if i == 0 else True))
                         for i, (part, idx) in enumerate(parts)]
                    numbers += cls._get_numbers(s)
        return numbers

    NUM_RE = {
        'conventional': r'[+-]?\d+(?:\.\d*)?%?',
        'comma': r'[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?',
        'decimal': r'[+-]?\.\d+%?',
    }

    @classmethod
    def _is_legal(cls, seg: str, category='conventional'):
        if category == 'simple':
            return any(cls._is_legal(seg, key) for key in cls.NUM_RE.keys())
        elif category in cls.NUM_RE:
            matches = re.findall(cls.NUM_RE[category], seg)
            # print("Matches: ", seg, category, matches)
            if len(matches) == 0:
                return False
            else:
                return matches[0] == seg
        else:
            raise ValueError(f"{category} is not a valid category.")

    @staticmethod
    def _split_seps(segment: str, seg_start: int = 0, sep: str = ',') -> List[Tuple[str, int]]:
        parts = re.split(f"[{sep}]", segment)
        part_start = seg_start
        results = []
        for part in parts:
            results.append((part, part_start))
            part_start += len(part) + 1

        while len(results) > 0:
            if results[0][0] == "":  # ignore first empty part
                results = results[1:]
            elif results[-1][0] == "":  # ignore last empty part
                results = results[:-1]
            else:
                break
        return results

    @classmethod
    def _is_comb(cls, seg: str):
        for sep in cls.SEP_CHARS:
            parts = cls._split_seps(seg, sep=sep)
            # print(f"Parts({sep}): ", seg, parts)
            if all([cls._is_legal(p, 'simple') for p, _ in parts]):
                return sep
        return None

    @classmethod
    def _is_covered(cls, seg: str):
        if cls._is_legal(seg, 'conventional'):
            return True
        if cls._is_legal(seg, 'comma'):
            return True
        if cls._is_legal(seg, 'decimal'):
            return True
        if cls._is_comb(seg):
            return True
        return False


HF_MODEL_NAMES = {
    "tapas": "google/tapas-base-masklm",
    "roberta": "roberta-large",
    "bert": "bert-large-uncased"
}

HF_TOKENIZERS = {
    "tapas": TapasTokenizer,
    "roberta": RobertaTokenizer,
    "bert": BertTokenizer
}


def prepare_tokenizer(model_name: str, model_dir: str = "/storage/tuna-models/",
                      num_token: str = "[NUM]", redirect_huggingface_cache=1) -> Tuple[PreTrainedTokenizer, str]:
    if model_name == "roberta":  # Temporary work-around for roberta model downloaded from AWS
        tokenizer = RobertaTokenizer.from_pretrained(os.path.join(model_dir, "roberta.large"))
    elif model_name not in HF_MODEL_NAMES:
        raise ValueError(f"{model_name} tokenizer not supported.")
    else:
        cache_dir = os.path.join(model_dir, "huggingface") if redirect_huggingface_cache else None
        tokenizer = HF_TOKENIZERS[model_name].from_pretrained(HF_MODEL_NAMES[model_name], cache_dir=cache_dir)

    assert tokenizer.add_tokens([num_token]) == 1  # Add [NUM] token to vocab.
    tokenizer.num_token_id = len(tokenizer) - 1

    return tokenizer


def unpack_and_pad_number_property(batch_property, device, padding_value=0, batch_first=True):
    unpacked = list(map(lambda property_for_single_number:
                        torch.tensor(list(map(int, property_for_single_number.split('|')))).long(), batch_property))
    padded = pad_sequence(unpacked, batch_first=batch_first, padding_value=padding_value).to(device)
    return padded


def get_number_digit_mapping_from_number_list(number_list: List[str], single_side_digit_upper_bound=10):
    '''
    Used for mapping each digit to the positional embedding subscripts
    0: not a number character (e.g. the comma sign, the decimal dot or the plus/minux signs)
    [1, single_side_digit_upper_bound]: the ones place/tens place/hundreds place/etc
    [single_side_digit_upper_bound+1, single_side_digit_upper_bound*2]: the tenths place/hundredths place/etc
    '''
    number_digit_mappings = []
    for number in number_list:
        mapping = [0] * len(number)
        # Get number_digit_mapping for the number
        if '.' in number:
            dot_idx = number.index('.')
        else:
            dot_idx = len(number)
        int_digits = sum([_ in DIGIT_CHARS for _ in number[:dot_idx]])
        frac_digits = single_side_digit_upper_bound + 1
        for idx, char in enumerate(number):
            if char in DIGIT_CHARS:
                if idx < dot_idx:
                    mapping[idx] = min(single_side_digit_upper_bound, int_digits)
                    int_digits -= 1
                else:
                    mapping[idx] = min(single_side_digit_upper_bound * 2, frac_digits)
                    frac_digits += 1
        mapping_string = '|'.join(map(str, mapping))
        number_digit_mappings.append(mapping_string)
    return number_digit_mappings


def get_tuta_feat(number_list: List[str]):
    tuta_feats = []
    num_chars = DIGIT_CHARS
    for number in number_list:
        if '.' in number:
            mag = str(min(sum([_ in num_chars for _ in number.split('.')[0]]), 9))
            prec = str(min(sum([_ in num_chars for _ in number.split('.')[0]]), 9))
        else:
            mag = str(min(sum([_ in num_chars for _ in number]), 9))
            prec = '0'
        msd = [_ for _ in number if _ in num_chars][0]
        lsd = [_ for _ in number if _ in num_chars][-1]
        tuta_feats.append('|'.join([mag, prec, msd, lsd]))
    return tuta_feats


def get_format_feat(number_list: List[str], length_upper_bound=12):
    batch_length = []
    batch_int = []
    batch_negative = []
    batch_percentage = []
    num_chars = DIGIT_CHARS
    for number in number_list:
        batch_length.append(min(sum([_ in num_chars for _ in number]), length_upper_bound))
        batch_int.append(int('.' not in number))
        batch_negative.append(int('-' in number))
        batch_percentage.append(int('%' in number))

    return batch_length, batch_int, batch_negative, batch_percentage


if __name__ == '__main__':
    texts = [
        '30 June 2018',
        '16,284',
        '1,234,567,890.9999',
        '15.67%, 30.25% %%',
        '(1,746)',
        '1.5, 2., .3,',
        '1234,5678,90123',
        '192.168.25.3',
        '60-1000',
        '192+180=372',
        '+1000% -50',
        'USD -100',
        '$-1000',
        '-$100',
        '-USD100',
        'F-1 Format',
        'Boeing-737',
        'https://arxiv.org/abs/1809.08887',
        '8533baebba27067f05b73c760b332112',
        '15.67%, 9,123,430.25% %% ,988, 1234,5567,789',
        '-10%,+15.423%, -.5% ,+10.%, 0.99 ,123,456,789, +-10',
        '192.168.2.34.',
        ',,,,123,567,890,,,,,,'
    ]
    for t in texts:
        print(f'"{t}" is parsed into ', NumTok.find_numbers(t))
        # Output:
        # 30 June 2018 is parsed into  [('30', 0, 2), ('2018', 8, 12)]
        # 16,284 is parsed into  [('16,284', 0, 6)]
        # (1,746) is parsed into  [('1,746', 1, 6)]

    texts = [
        '+.7%',
        '16,284.1231',
        '1203',
        '1,234,567.34%',
        '-2147483648',
        '-5.67%'
    ]
    for t in texts:
        sig, exp = NumTok.get_sci(t)
        print(f'"{t}": raw value = {NumTok.get_val(t)}, significand = {sig}, exponent = {exp}')
        # Output:
        # The raw float value of +.7% is  0.006999999999999999
        # The raw float value of 16,284.1231 is  16284.1231
        # The raw float value of 1203 is  1203.0
