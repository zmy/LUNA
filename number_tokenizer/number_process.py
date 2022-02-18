import torch

from number_tokenizer.vocab import NumVocab
from .numtok import NumTok as nt

EXP_MIN, EXP_MAX = -2, 7


def full_preprocess(number_list):
    number, sig, exp = number_list
    number_string = str(number)
    token_ids = [NumVocab.char2idx(_) for _ in number_string]

    # Get tokenization (e.g. 42 -> ['2', [DEC], '4', [DEC]])
    decs = [NumVocab.DEC for _ in number_string]
    token_id_with_dec = [None] * 2 * len(decs)
    token_id_with_dec[::2] = token_ids[::-1]
    token_id_with_dec[1::2] = decs

    token_id_with_dec = torch.tensor(token_id_with_dec).long()
    sig_gt = torch.tensor(float(sig))
    exp_gt = torch.tensor(exp + 2).long()
    val_gt = torch.tensor(float(number))

    return token_id_with_dec, sig_gt, exp_gt, val_gt, number_string


def preprocess_with_dec_only(number_list):
    number, sig, exp = number_list
    number_string = str(number)
    token_ids = [NumVocab.char2idx(_) for _ in number_string]

    # Insert decs into token_ids
    decs = [NumVocab.DEC for _ in number_string]
    token_id_with_dec = [None] * 2 * len(decs)
    token_id_with_dec[::2] = token_ids
    token_id_with_dec[1::2] = decs

    token_id_with_dec = torch.tensor(token_id_with_dec).long()
    sig_gt = torch.tensor(float(sig))
    exp_gt = torch.tensor(exp + 2).long()
    val_gt = torch.tensor(float(number))

    return token_id_with_dec, sig_gt, exp_gt, val_gt, number_string


def preprocess_with_reverse_only(number_list):
    number, sig, exp = number_list
    number_string = str(number)
    token_ids = [NumVocab.char2idx(_) for _ in number_string]

    # reverse token_ids
    reversed_token_ids = [None] * len(token_ids)
    reversed_token_ids[::-1] = token_ids

    token_id_with_dec = torch.tensor(reversed_token_ids).long()
    sig_gt = torch.tensor(float(sig))
    exp_gt = torch.tensor(exp + 2).long()
    val_gt = torch.tensor(float(number))

    return token_id_with_dec, sig_gt, exp_gt, val_gt, number_string


def trivial_preprocess(number: str):
    sig, exp = nt.get_sci(number)
    exp = max(EXP_MIN, exp)
    exp = min(EXP_MAX, exp)
    token_ids = [NumVocab.char2idx(_) for _ in number]
    token_ids = torch.tensor(token_ids).long()

    sig_gt = torch.tensor(sig, dtype=torch.float32)
    exp_gt = torch.tensor([exp - EXP_MIN]).long()
    val_gt = torch.tensor(nt.get_val(number), dtype=torch.float32)

    return token_ids, sig_gt, exp_gt, val_gt, number


NUM_PROCESS_FUNCS = {
    'full': full_preprocess,
    'dec': preprocess_with_dec_only,
    'reverse': preprocess_with_reverse_only,
    'trivial': trivial_preprocess,
}
