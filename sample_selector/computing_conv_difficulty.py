from tqdm import tqdm
from IPython import embed
import numpy as np
import json
import copy
import pickle

train_file = '/data/train.json'
with open(train_file, 'r') as f:
    train_data = json.load(f)
sid2conv = {}
conv_lens = {}
for conv in tqdm(train_data):
    cnt = 0
    history = []
    for ind, turn in enumerate(conv['turns']):
        question = turn['question']
        response = turn['response']
        cnt+=1
        sample_id = "{}_{}".format(conv['conv_id'], turn['turn_id'])
        sid2conv[sample_id] = copy.deepcopy(history) + [question]
        history.append(question)
        history.append(response)
    conv_lens[str(conv['conv_id'])] = cnt

topic_cnts_file = '/data/topic_cnts.pkl'
with open(topic_cnts_file, 'rb') as file:
    topic_cnts = pickle.load(file)
    
ppl_path = '/data/ppl.jsonl'
ppls = {}
with open(ppl_path, 'r') as file:
    for line in file:
        json_line = json.loads(line.strip())
        ppls[json_line['sample_id']] = json_line['ppl']

avg_ppls = {}
for sample_id, ppl in ppls.items():
    conv_id, turn_id = sample_id.split('_')
    if conv_id in avg_ppls:
        avg_ppls[conv_id].append(ppl)
    else:
        avg_ppls[conv_id] = [ppl]
for sample_id, ppl in avg_ppls.items():
    avg_ppls[sample_id] = sum(ppl) / len(ppl)

sid2difficulty = {}
for sample_id in list(ppls.keys()):
    conv_id, turn_id = sample_id.split('_')
    conv_len = conv_lens[conv_id]
    avg_ppl = avg_ppls[conv_id]
    topic_cnt = topic_cnts[sample_id][0]
    sid2difficulty[sample_id] = topic_cnt*avg_ppl+conv_len

difficulty_file = '/data/conv_difficulty.pkl'
with open(difficulty_file, 'wb') as file:
    pickle.dump(sid2difficulty, file)