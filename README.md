# Building Persona Consistent Dialogue Agents with Offline Reinforcement Learning

Source code for our EMNLP submission "Building Persona Consistent Dialogue Agents with Offline Reinforcement Learning". We briefly describe our environment and the contents of each folder.

## Environment

All packages and versions used for our training and experiments are given in `requirements.txt`. We recommend placing our code and cloning the [ParlAI](https://github.com/facebookresearch/ParlAI) repo in your home directory so that the default paths in the config files will work by default. If you would like to download ParlAI directly in your environment instead, you can uncomment line 80 in `requirements.txt`.

 ## Data
 
To get the mapped dataset used for offline RL training you can simply run `get_mapped_data.sh`, this will download the PersonaChat and DNLI datasets and perform the mapping. The mapping process should take a little over an hour. We also provide the raw DNLI evaluation dataset. The dataset can also be downloaded [here](https://wellecks.com/dialogue_nli/).

## Offline_rl_bb3

The algorithm for performing offline rl training is found in `offline_rl_agent.py`. To recreate our offline rl training you can run `train.py`, setting `importance_sampling: varmi` will perform training using VaRMI importance sampling and `importance_sampling: gold` with perform training with gold importance sampling. All hyper-parameter settings can be found in `config1.yml`, any not listed are set to the default values in ParlAI. These values can be found in the ParlAI documentation [here](https://parl.ai/).

## Automatic_eval

To recreate our ranking evaluation you can run `python eval_ranking.py --data-path ~/dialogue-rienforce/data/eval_data`. You can change the value of `model_file` in the config file to evaluate different models. The DNLI evaluation set includes three different persona-candidate categories. In our results we simply summed the results for each category.

## Human_eval

This folder gives our raw human evaluation scores for each model as well as the code for performing Bayesian Calibration. Before performing calibration the raw scores must be formatted by running `convert_format.py`.
