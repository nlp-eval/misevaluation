import re
import numpy as np
import os
import openai
import random
import pandas as pd
from tqdm import tqdm
import time
import argparse, sys

parser=argparse.ArgumentParser()

ENGINES = ['davinci-002', 'curie-001', 'babbage-001', 'ada-001']

def get_outputs(input_text, reference, n=1, pbar=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    responses = {
        'prompt' : [input_text] * n,
        'ref': [reference] * n,
        'agent' : list(range(n))
    }

    for engine in ENGINES:
        responses[engine] = []

        for i in range(n):
            responses[engine].append(openai.Completion.create(
                    engine=f"text-{engine}",
                    prompt="Write a story based on this prompt.\n\nPrompt: " + input_text + "\n\nStory:",
                    temperature=0.9,
                    max_tokens=1600,
                    top_p=1,
            ).choices[0]['text'].strip())

            # progress bar
            if pbar is not None:
               pbar.update(1 / len(ENGINES))

            # rate limit
            time.sleep(1)

    return responses


def clean(text):
    text = re.search('(\[ WP \]\s?)*(.+)', text).group(2)
    text = re.sub(r'\s?<newline>\s?', '\n', text)
    text = re.sub(r"\s+([?\'.,!\"]|n't)", r'\1', text)
    text = re.sub(r"``\s+", "``", text)
    return text.strip()


def create_dataset(source_fn, target_fn, size=1, num_agents=1, seed=1):
    """
    size: number of examples per person
    """
    random.seed(seed)
    dfs = []
    indices = []

    print("Filtering examples ...")
    indices = [ i for i,x in enumerate(open(source_fn)) if x.startswith('[ WP ]') ]
    sampled = random.sample(indices, min(size, len(indices)))

    print("Querying GPT3 ...")

    with open(source_fn) as source, open(target_fn) as target:
        pbar = tqdm(total=(len(sampled) * num_agents))
        i = 0 
        
        for x,y in tqdm(zip(source, target)):
            if i in sampled: 
                dfs.append(pd.DataFrame.from_dict(
                    get_outputs(clean(x), clean(y), n=num_agents, pbar=pbar)
                ))

            i += 1

        pbar.close()

    return pd.concat(dfs)


if __name__ == "__main__":
    parser.add_argument('--prompt_fn', help='prompt file', required=True, type=str)
    parser.add_argument('--ref_fn', help='human references file', required=True, type=str)
    parser.add_argument('--size', help='number of samples per agent', default=1, type=int)
    parser.add_argument('--num_agents', help='number of agents', default=1, type=int)
    parser.add_argument('--out_fn', help='output file', required=True, type=str)
    args = parser.parse_args()

    table = create_dataset(args.prompt_fn, args.ref_fn, size=args.size, num_agents=args.num_agents)
    table.to_csv(args.out_fn, index=False)

