from enum import IntEnum


class NumVocab(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    DEC = 10
    NEGATION = 11
    SCI = 12
    DOT = 13
    PERCENT = 14
    COMMA = 15
    PLUS = 16
    PAD = 17
    UNK = 18

    @staticmethod
    def char2idx(char: str):
        if char in SPECIAL_NUM_CHARS:
            return SPECIAL_NUM_CHARS[char]
        elif ord('0') <= ord(char) <= ord('9'):
            return NumVocab(ord(char) - ord('0'))
        else:
            raise ValueError(f"Unexpected char: '{char}' -- not in vocab.")
            # return NumVocab.UNK  TODO: enumerate all possible chars

    @classmethod
    def size(cls):
        return len(cls)


SPECIAL_NUM_CHARS = {
    '.': NumVocab.DOT,
    '-': NumVocab.NEGATION,
    '%': NumVocab.PERCENT,
    ',': NumVocab.COMMA,
    '+': NumVocab.PLUS
}

UNTIDY = [
    ',',
    '%'
]

DIGIT_ENUM = [
    NumVocab.ZERO,
    NumVocab.ONE,
    NumVocab.TWO,
    NumVocab.THREE,
    NumVocab.FOUR,
    NumVocab.FIVE,
    NumVocab.SIX,
    NumVocab.SEVEN,
    NumVocab.EIGHT,
    NumVocab.NINE,
]

DIGIT_CHARS = '0123456789'
