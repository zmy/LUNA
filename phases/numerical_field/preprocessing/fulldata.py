from google.protobuf import text_format
from tapas.protos import interaction_pb2
import json

tables={}
texts={}
pairs=[]
with open('interactions.txtpb',encoding='utf8') as f:
    i=0
    while True:
        line=f.readline()
        if line=='':break
        try:
            interaction=text_format.Parse(line, interaction_pb2.Interaction())
            table=interaction.table
            _table=[]
            _table.append([column.text for column in table.columns])
            for row in table.rows:
                _table.append([cell.text for cell in row.cells])
            _text=interaction.questions[0].original_text+interaction.questions[1].original_text
            tables[str(i)]=_table
            texts[str(i)]=_text
            pairs.append((str(i),str(i)))
            i += 1
            if i%10000==0:
                print(i)
        except:
            pass

with open('all_table.json','w',encoding='utf8')as f:
    json.dump(tables,f,indent=4)
with open('all_text.json','w',encoding='utf8')as f:
    json.dump(texts,f,indent=4)
with open('all_pair.json','w',encoding='utf8')as f:
    json.dump(pairs,f,indent=4)