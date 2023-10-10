import argparse
import json
import jsonlines
import yaml
import numpy as np
from collections import Counter
from tqdm import tqdm
from parlai.core.agents import create_agent_from_model_file, create_agent_from_opt_file

import random

def eval_model(eval_data):
    choices = []
    for item in tqdm(eval_data):
        message = {'text': '\n'.join(['your_persona: ' + i for i in item['persona']])+'\n'+'\n'.join(item['prefix']), 
                'episode_done': True}
        best_loss = np.inf
        best_candidate = ''
        # phrase = ''
        for k in item['candidates'].keys():
            for candidate in item['candidates'][k]:
                message['eval_labels'] = [candidate]
                blender_agent.observe(message)
                response = blender_agent.act()
                if response['metrics']['loss'].value() < best_loss:
                    best_loss = response['metrics']['loss'].value()
                    best_candidate = k
                    # phrase = candidate
        
        choices.append(best_candidate)
    
    return choices

def main(args):
    attr_eval = []
    with jsonlines.open(args.data_path + '/valid_attributes.jsonl', 'r') as reader:
        for obj in reader:
            attr_eval.append(obj)

    haves_eval = []
    with jsonlines.open(args.data_path + '/valid_havenot.jsonl', 'r') as reader:
        for obj in reader:
            haves_eval.append(obj)

    likes_eval = []
    with jsonlines.open(args.data_path + '/valid_likedislike.jsonl', 'r') as reader:
        for obj in reader:
            likes_eval.append(obj)

    with open(args.config_path, 'r') as stream:
        opt = yaml.safe_load(stream)
    

    global blender_agent
    blender_agent = create_agent_from_opt_file(opt)

    res = eval_model(attr_eval)
    print(f'Attributes eval:')
    print(Counter(res))
                
    res = eval_model(haves_eval)
    print(f'Haves eval:')
    print(Counter(res))

    res = eval_model(likes_eval)
    print(f'Likes eval:')
    print(Counter(res))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config.yml",
        help="path to config file")
    parser.add_argument("--data-path", type=str,
        help="path to eval data folder")
    args = parser.parse_args()
    main(args)