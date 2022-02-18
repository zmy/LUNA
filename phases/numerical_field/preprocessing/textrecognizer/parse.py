import os
from tqdm import tqdm
import json
from collections import defaultdict
input_dir=r"C:\Users\v-hongweihan\Downloads\NumberDataset"
id=0
text=[]
statistics=defaultdict(lambda:{'table':[],'text':[]})
print('parsing TabFact tables')
path=os.path.join(input_dir,r"Table-Fact-Checking\data\all_csv")
for x in tqdm(sorted(os.listdir(path))):
    if not x.endswith("csv"):continue
    fin=open(os.path.join(path,x),encoding="utf8")
    fout=open(os.path.join(path, x+'.num'),'w',encoding="utf8")
    for line in fin.readlines():
        row=line.strip().split('#')
        text.extend(row)
        statistics['TabFact']['table'].extend(list(range(id,id+len(row))))
        fout.write('%s\n'%('#'.join([cell+"="+str(_) for cell,_ in zip(row,range(id,id+len(row)))])))
        id+=len(row)
    fin.close()
    fout.close()

print('parsing TabFact captions')
path=os.path.join(input_dir,r"Table-Fact-Checking\tokenized_data")
for x in tqdm(sorted(os.listdir(path))):
    if not x.endswith("json"):continue
    with open(os.path.join(path, x),encoding="utf8")as f:
        split=json.load(f)
    for v in split.values():
        text.extend(v[0])
        statistics['TabFact']['text'].extend(list(range(id,id+len(v[0]))))
        for i in range(len(v[0])):
            v[0][i]+='='+str(id)
            id+=1
        text.append(v[2])
        statistics['TabFact']['text'].append(id)
        v[2]+='='+str(id)
        id+=1
    with open(os.path.join(path, x+'.num'),'w',encoding="utf8") as f:
        json.dump(split,f,indent=4)

print('parsing TATQA')
path=os.path.join(input_dir,r"TAT-QA")
for x in tqdm(sorted(os.listdir(path))):
    if not x.endswith("json"):continue
    with open(os.path.join(path, x),encoding="utf8")as f:
        split=json.load(f)
    for v in split:
        table=v['table']['table']
        for col in table:
            text.extend(col)
            statistics['TATQA']['table'].extend(list(range(id, id + len(col))))
            for i in range(len(col)):
                col[i]+='='+str(id)
                id+=1
        parag=v['paragraphs']
        for p in parag:
            text.append(p['text'])
            statistics['TATQA']['text'].append(id)
            p['text']+='='+str(id)
            id+=1
        questions=v['questions']
        for q in questions:
            text.append(q['question'])
            statistics['TATQA']['text'].append(id)
            q['question']+='='+str(id)
            id+=1
    with open(os.path.join(path, x+'.num'),'w',encoding="utf8") as f:
        json.dump(split,f,indent=4)

print('parsing WTQ tables')
path = os.path.join(input_dir, r"WikiTableQuestions\csv")
for dir in sorted(os.listdir(path)):
    for x in tqdm(sorted(os.listdir(os.path.join(path,dir)))):
        if not x.endswith("tsv"): continue
        fin = open(os.path.join(path,dir, x), encoding="utf8")
        fout = open(os.path.join(path,dir, x + '.num'), 'w', encoding="utf8")
        for line in fin.readlines():
            row = line.strip().split('\t')
            text.extend(row)
            statistics['WTQ']['table'].extend(list(range(id, id + len(row))))
            fout.write('%s\n' % ('\t'.join([cell + "=" + str(_) for cell, _ in zip(row, range(id, id + len(row)))])))
            id += len(row)
        fin.close()
        fout.close()

print('parsing WTQ questions')
path = os.path.join(input_dir, r"WikiTableQuestions\data")
for x in tqdm(sorted(os.listdir(path))):
    if not x.endswith("tsv"):continue
    fin = open(os.path.join(path, x), encoding="utf8")
    fout = open(os.path.join(path, x + '.num'), 'w', encoding="utf8")
    for i,line in enumerate(fin.readlines()):
        if i==0:
            fout.write(line)
            continue
        row = line.strip().split('\t')
        text.append(row[1])
        statistics['WTQ']['text'].append(id)
        row[1]+="=" + str(id)
        fout.write('%s\n' % ('\t'.join(row)))
        id += 1
    fin.close()
    fout.close()

with open(os.path.join(input_dir,'text.json'),'w',encoding='utf8')as f:
    json.dump(text,f,indent=4)
with open(os.path.join(input_dir,'statistics.json'),'w',encoding='utf8')as f:
    json.dump(statistics,f,indent=4,sort_keys=True)
