from sentence_transformers import SentenceTransformer as SBert, util
from tqdm import tqdm
import os
import re
import json
import pickle
from itertools import combinations
import numpy as np
from IPython import embed

from itertools import combinations
from tqdm import tqdm
import numpy as np
import torch

model = SBert('all-MiniLM-L6-v2')

train_file = '/data/train.json'
with open(train_file, 'r') as f:
    train_data = json.load(f)
print(len(train_data))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(device)

def compute_batch_sim(sent_pairs, score_lists):
    sents1, sents2 = zip(*sent_pairs)
    embeddings1 = model.encode(list(sents1), convert_to_tensor=True).to(device)
    embeddings2 = model.encode(list(sents2), convert_to_tensor=True).to(device)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    for i, score in enumerate(cosine_scores):
        score_lists[i].append(score[i].cpu().numpy())

def remove_pairs(run_pairs, no_pairs):
    new_pairs = [pair for pair in run_pairs if pair not in no_pairs and pair[::-1] not in no_pairs]
    return new_pairs

augs = ['token_deletion', 'turn_deletion', 'reorder', 'turn_deletion_depend', 'reorder_depend', 'paraphrase', 'extend', 'swap', 'shift']
no_pairs = [('turn_deletion', 'turn_deletion_depend'), ('reorder', 'reorder_depend'), ('turn_deletion', 'reorder_depend'), ('turn_deletion_depend', 'reorder'), ('swap', 'shift')]
run_pairs = list(combinations(augs, 2))
run_pairs = remove_pairs(run_pairs, no_pairs)
sims = {aug1+'+'+aug2: [] for (aug1, aug2) in run_pairs}

batch_size = 512

for conv in tqdm(train_data):
    history = []
    batch_pairs = []
    batch_score_lists = []

    for ind, turn in enumerate(conv['turns']):
        if len(turn["pos_doc_ids"]) == 0:
            history.append(turn['question'])
            history.append(turn['response'])
            continue

        aug_ctx = turn['aug_ctx']
        
        for (aug1, aug2) in run_pairs:
            if aug1 not in aug_ctx or aug2 not in aug_ctx:
                continue
            sent1 = " ".join(aug_ctx[aug1])
            sent2 = " ".join(aug_ctx[aug2])
            batch_pairs.append((sent1, sent2))
            batch_score_lists.append(sims[aug1+'+'+aug2])

            if len(batch_pairs) >= batch_size:
                compute_batch_sim(batch_pairs, batch_score_lists)
                batch_pairs = []
                batch_score_lists = []

        history.append(turn['question'])
        history.append(turn['response'])

    if batch_pairs:
        compute_batch_sim(batch_pairs, batch_score_lists)

for (aug1, aug2) in run_pairs:
    print("{} \t\t Score: {:.4f}".format(aug1+'+'+aug2, np.mean(sims[aug1+'+'+aug2])))


sim_file = '/data/sims_aug_pair.pkl'
with open(sim_file, 'wb') as file:
    pickle.dump(sims, file)