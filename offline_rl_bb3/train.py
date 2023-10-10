import argparse
import yaml
import json
import torch
import random
from tqdm import tqdm
from more_itertools import batched
from parlai.core.agents import create_agent_from_model_file, create_agent_from_opt_file

def test_loss(data, blender_agent):
    positive_loss = []
    negative_loss = []

    for batch in batched(tqdm(data), opt['eval_batchsize']):
        observations = []
        rewards = []
        for msg in batch:
            rewards.append(msg['reward'])
            message = {'text': '\n'.join(msg['persona_other'])+'\n'+'\n'.join(msg['persona_self'])+'\n'+'\n'.join(msg['context']), 
                        'eval_labels': [msg['label']], 'episode_done': True}
            observations.append(blender_agent.observe(message))
            blender_agent.reset()
        response = blender_agent.batch_act(observations)
        for i in range(len(response)):
            if rewards[i] == '-1':
                negative_loss.append(response[i]['metrics']['loss'].value())
            else:
                positive_loss.append(response[i]['metrics']['loss'].value())
    
    print(f'positive loss: {sum(positive_loss)/len(positive_loss)}')
    print(f'negative loss: {sum(negative_loss)/len(negative_loss)}')
    return sum(positive_loss)/len(positive_loss), sum(negative_loss)/len(negative_loss)

def main(args):
    with open(args.config_path, 'r') as stream:
        global opt
        opt = yaml.safe_load(stream)

    with open(opt['training_data']) as stream:
        data = json.load(stream)

    blender_agent = create_agent_from_opt_file(opt)

    random.Random(4).shuffle(data)
    train = data[:-5000]
    test = data[-5000:]

    with torch.no_grad():
        print('test loss:')
        test_loss(test, blender_agent)

    # custom training loop
    for epoch in range(opt['num_epoch']):
        for batch in batched(tqdm(train), opt['batchsize']):
            observations = []
            for msg in batch:
                message = {'text': '\n'.join(msg['persona_other'])+'\n'+'\n'.join(msg['persona_self'])+'\n'+'\n'.join(msg['context']), 
                            'labels': [msg['label']], 'episode_done': True, 'reward':int(msg['reward'])}
                observations.append(blender_agent.observe(message))
                blender_agent.reset()
            response = blender_agent.batch_act(observations)
            # print('act')
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            print(f'epoch: {epoch}')
            pl, nl = test_loss(test, blender_agent)
        
        print('saving model...')
        blender_agent.save(opt['save_model_path']+'/checkpoint'+str(epoch))
        with open(opt['save_model_path']+f'/checkpoint{epoch}_loss.txt', 'w') as f:
            f.write(f"positive_loss: {pl}\n")
            f.write(f"negative_loss: {nl}\n")
        print('model_saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config1.yml",
        help="path to config file")
    args = parser.parse_args()
    main(args)