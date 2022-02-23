'''
this script is to generate prompt for column distribution
'''
import json
import random

from tqdm import tqdm

from ..dataset.utils import parse_string, compute_col_distribution

with open('/mnt/tmp_data/all_table.json', encoding='utf8') as f:
    tables = json.load(f)
pairs = []
texts = {}
prefix = 'extra'
id = 0


def func(table):
    if len(table) < 3 or len(table[0]) == 0: return []
    col = []
    for j, row in enumerate(table):
        for i in range(len(row)):
            new_text, _ = parse_string(row[i], 128, 3)
            if j == 0:
                col.append([])
            else:
                if len(_) == 1:
                    col[i].append(_[0][1])
                else:
                    col[i].append(None)
    ret = []
    for col_id, (col_name, x) in enumerate(zip(table[0], col)):
        distribution = compute_col_distribution(x, ret_sum=True)
        if distribution[0] < -5e10: continue
        fmts = ['%.2f'] * 6
        feats = [random.sample(['minimal', 'smallest', 'lowest'], 1)[0], \
                 random.sample(['maximal', 'largest', 'highest'], 1)[0], \
                 random.sample(['average', 'mean'], 1)[0], \
                 random.sample(['ratio of value between 0 and 1', 'ratio of 0~1 range'], 1)[0], \
                 random.sample(['ratio of value between 0 and 100', 'ratio of 0~100 range'], 1)[0], \
                 random.sample(['sum', 'total'], 1)[0]]
        for i in [0, 1, 5]:
            if round(distribution[i]) == distribution[i]:
                distribution[i] = round(distribution[i])
                fmts[i] = '%d'
        if col_name == '':
            col_name = 'column %d' % (col_id + 1)
        for feat, val, fmt in zip(feats, distribution, fmts):
            ret.append(random.sample([ \
                ('the %s of %s is ' + fmt) % (feat, col_name, val), \
                ('%s has the %s: ' + fmt) % (col_name, feat, val), \
                (fmt + ' is the %s of %s') % (val, feat, col_name) \
                ], 1)[0])
    return ret


for name, table in tqdm(tables.items()):
    _texts = func(table)
    for _text in _texts:
        texts['%s/%d' % (prefix, id)] = _text
        pairs.append((name, '%s/%d' % (prefix, id)))
        id += 1
with open('/mnt/tmp_data/extra_text.json', 'w', encoding='utf8') as f:
    json.dump(texts, f)
with open('/mnt/tmp_data/extra_pair.json', 'w', encoding='utf8') as f:
    json.dump(pairs, f)
