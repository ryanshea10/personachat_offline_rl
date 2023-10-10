import json
import jsonlines
from tqdm import tqdm
from transformers import pipeline
from projects.bb3.agents import utils

import logging
logging.disable(logging.CRITICAL)

# contradiction classifier for filtering personas

classifier = pipeline('text-classification', model='roberta-large-mnli', device=0)


# read data

with open('aligned_data.json') as f:
    data = json.load(f)

dnli = []
with jsonlines.open('raw_data/dnli/dialogue_nli/dialogue_nli_train.jsonl', 'r') as reader:
    for obj in reader:
        for i in obj:
            dnli.append(i)

dnli_sents = {}
for s in dnli:
    dnli_sents[s['sentence1']] = s['triple1']
    dnli_sents[s['sentence2']] = s['triple2']

format_correction = {'i ve reached 50k subscribers ! .': 'i ve reached 50k subscribers !',
                     'i have reached 50k subscribers ! .': 'i have reached 50k subscribers !',
                     'i love cooking ! .': 'i love cooking !'}


# partner has first turn, align convo so the label matches "your persona"

for d in data:
    if len(d['context']) % 2 == 0:
        p1_swap = ["partner's persona: " + i[14:] for i in d['persona_self']]
        p2_swap = ["your persona: " + i[19:] for i in d['persona_other']]

        if len(d['context']) == 0:
            d['context'] = ['hi']
        else:
            d['context'] = d['context'][1:]
        d['persona_self'] = p2_swap
        d['persona_other'] = p1_swap


# make sure the persona isn't contradictory

for d in tqdm(data):
    persona = [d['candidate']]
    label = d['label']
    for item in d['persona_self']:
        persona_format = item[14:-1] + " " + item[-1]
        if persona_format in format_correction.keys():
            persona_format = format_correction[persona_format]
        
        persona_r2 = dnli_sents[persona_format][-1]
        if persona_r2 not in label and persona_r2 not in dnli_sents[label] and utils.no_overlap(persona, persona_format):
            if 'employed_by_general' in dnli_sents[label] and 'employed_by_general' in dnli_sents[persona_format]:
                continue
            if d['reward'] == '-1' and classifier(persona_format.replace(', ', '') + ' , ' + d['candidate'])[0]['label'] == 'CONTRADICTION':
                continue
            persona.append(item)

    persona[0] = 'your persona: ' + d['candidate']
    d['persona_self'] = persona.copy()

with open('filtered_data.json', 'w') as f:
    json.dump(data, f, indent = 4)

