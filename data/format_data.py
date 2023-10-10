import json
import jsonlines
from tqdm import tqdm


# read data

with open('raw_data/personachat/train_both_original.txt', 'r') as inp:
    personachat = inp.readlines()

dnli = []
with jsonlines.open('raw_data/dnli/dialogue_nli/dialogue_nli_train.jsonl', 'r') as reader:
    for obj in reader:
        for i in obj:
            dnli.append(i)


# get phrases

personas = set()
pc = []
for i in tqdm(range(len(personachat))):
    line = personachat[i][:personachat[i].find('\t\t')]
    if '1 your persona' not in line:
        line = line[line.find(' ')+1:]
    pc += line.split('\t')

    if 'persona: ' in line:
        personas.add(line[line.find(": ")+2:-1])


# get indexes of matching dialogues

total = 0
idxs = []
for pairs in tqdm(dnli):
    if pairs['label'] == 'neutral':
        continue
    s1 = pairs['sentence1']
    s2 = pairs['sentence2']
    rew = 1 if pairs['label'] == 'positive' else -1


    try:
        idx = pc.index(s1)
        if s2[:-2] in personas:
            idxs.append((idx,s2,rew))
            total += 1
            continue
    except ValueError:
        pass
    except KeyboardInterrupt:
        break
    
    try:
        idx = pc.index(s2)
        if s1[:-2] in personas:
            idxs.append((idx,s1,rew))
            total += 1
            continue
    except ValueError:
        pass
    except KeyboardInterrupt:
        break


# match personas

for i in idxs:
    pc[i[0]] = i[1] + '\t' + str(i[2]) + '\t\t' + pc[i[0]]


# align data (unfiltered)

all_data = []
data = {}
persona1 = []
persona2 = []
context = []
for line in pc:
    if line[:16] == '1 your persona: ':
        data = {}
        persona1 = [line[2:]]
        persona2 = []
        context = []
        continue

    if 'your persona: ' in line:
        persona1.append(line)
        continue
    elif "partner's persona: " in line:
        persona2.append(line)
        continue
    
    if '\t\t' not in line:
        context.append(line)
    else:
        phrases = line.split('\t\t')
        l = len(phrases)
        for i, phrase in enumerate(phrases):
            if i+1 < l:
                candidate, rew = phrase.split('\t')
                data['persona_self'] = persona1
                data['persona_other'] = persona2
                data['context'] = context.copy()
                data['candidate'] = candidate
                data['reward'] = rew
                data['label'] = phrases[-1]
                all_data.append(data.copy())
        
        context.append(phrases[-1])


# write data

with open('aligned_data.json', 'w') as fout:
    json.dump(all_data, fout, indent = 4)