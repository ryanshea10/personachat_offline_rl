import argparse
import numpy as np

def main(args):
    with open(f'scores/base/{args.metric}.txt', 'r') as f:
        base = f.readlines()

    with open(f'scores/gold/{args.metric}.txt', 'r') as f:
        gold = f.readlines()

    with open(f'scores/varmi/{args.metric}.txt', 'r') as f:
        varmi = f.readlines()

    combined = zip(base,gold,varmi)

    dta = []
    i = 0
    for row in combined:
        for score in row:
            if i % 3 == 0:
                dta.append([0, i, int(score)])
            elif i % 3 == 1:
                dta.append([1, i, int(score)])
            else: 
                dta.append([2, i, int(score)])
            i += 1

    np_dta = np.asarray(dta)

    with open(f'{args.metric}.npy', 'wb') as f:
        np.save(f, np_dta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, choices=['quality', 'rep_persona'],
        help="path to config file")
    args = parser.parse_args()
    main(args)