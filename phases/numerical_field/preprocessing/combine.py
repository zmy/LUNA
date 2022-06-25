import os
from tqdm import tqdm
import json
from collections import defaultdict
input_dir=r"<your raw data path>"
table_id=0
text_id=0
tables={}
texts={}
pairs=[]

print('parsing TabFact tables')
path=os.path.join(input_dir,r"Table-Fact-Checking\data\all_csv")
for x in tqdm(sorted(os.listdir(path))):
    if not x.endswith("csv"):continue
    fin=open(os.path.join(path,x),encoding="utf8")
    table= []
    for i,line in enumerate(fin.readlines()):
        row=line.strip().split('#')
        table.append(row)
    fin.close()
    assert 'tabfact/%s'%x not in tables
    tables['tabfact/%s'%x]=table

print('parsing TabFact captions')
path=os.path.join(input_dir,r"Table-Fact-Checking\tokenized_data")
for x in tqdm(['train_examples.json']):
    if not x.endswith("json"):continue
    with open(os.path.join(path, x),encoding="utf8")as f:
        split=json.load(f)
    for k,v in split.items():
        for i,(cap,lab) in enumerate(zip(v[0],v[1])):
            if lab==1:
                assert 'tabfact/%s/%d'%(k,i) not in texts
                texts['tabfact/%s/%d'%(k,i)]=v[2]+' . '+cap
                pairs.append(('tabfact/%s'%k,'tabfact/%s/%d'%(k,i)))

print('parsing TATQA')
path=os.path.join(input_dir,r"TAT-QA")
for x in tqdm(sorted(os.listdir(path))):
    if not x.endswith("json"):continue
    with open(os.path.join(path, x),encoding="utf8")as f:
        split=json.load(f)
    for v in split:
        table=v['table']['table']
        _table=[]
        for i,col in enumerate(table):
            for j,cell in enumerate(col):
                if i==0:
                    _table.append([])
                _table[j].append(cell)
        assert 'tatqa/%s'%v['table']['uid'] not in tables
        tables['tatqa/%s'%v['table']['uid']]=_table

        parag=v['paragraphs']
        for p in parag:
            assert 'tatqa/%s'%p['uid'] not in texts
            texts['tatqa/%s'%p['uid']]=p['text']
            pairs.append(('tatqa/%s'%v['table']['uid'],'tatqa/%s'%p['uid']))
        questions=v['questions']
        for q in questions:
            assert 'tatqa/%s' % q['uid'] not in texts
            texts['tatqa/%s' % q['uid']] = q['question']
            pairs.append(('tatqa/%s' % v['table']['uid'], 'tatqa/%s' % q['uid']))

print('parsing WTQ tables')
path = os.path.join(input_dir, r"WikiTableQuestions\csv")
for dir in sorted(os.listdir(path)):
    for x in tqdm(sorted(os.listdir(os.path.join(path,dir)))):
        if not x.endswith("tsv"): continue
        fin = open(os.path.join(path,dir, x), encoding="utf8")
        lines = fin.readlines()
        fin.close()
        max_length = max([len(line.strip().split('\t')) for line in lines])
        table = []
        for i, line in enumerate(lines):
            row = line.strip().split('\t')
            table.append(row+[""]*(max_length-len(row)))
        assert 'wtq/csv/%s/%s' % (dir,x) not in tables
        tables['wtq/csv/%s/%s' % (dir,x)] = table

print('parsing WTQ questions')
path = os.path.join(input_dir, r"WikiTableQuestions\data")
for x in tqdm(sorted(os.listdir(path))):
    if not x.endswith("tsv"):continue
    fin = open(os.path.join(path, x), encoding="utf8")

    for i,line in enumerate(fin.readlines()):
        if i==0:
            continue
        row = line.strip().split('\t')
        assert 'wtq/%s' % row[0] not in texts
        texts['wtq/%s' % row[0]] = row[1]
        pairs.append(('wtq/%s' % (row[2][:-4]+'.tsv'), 'wtq/%s' % row[0]))

    fin.close()


with open(os.path.join(input_dir,'PretrainDataset','all_table.json'),'w',encoding='utf8')as f:
    json.dump(tables,f,indent=4)
with open(os.path.join(input_dir,'PretrainDataset','all_text.json'),'w',encoding='utf8')as f:
    json.dump(texts,f,indent=4)
with open(os.path.join(input_dir,'PretrainDataset','all_pair.json'),'w',encoding='utf8')as f:
    json.dump(pairs,f,indent=4)
