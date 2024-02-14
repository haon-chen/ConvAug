from tqdm import tqdm
from IPython import embed
import re
import json
import random

from itertools import combinations

def identify_swappable_turns(dependency_list):
    swappable_turns = []
    for i, j in combinations(range(len(dependency_list)), 2):
        can_swap = all(dep not in range(i + 1, j + 1) for dep in dependency_list[j])
        if can_swap:
            swappable_turns.append((i + 1, j + 1))
    return swappable_turns

train_file = './data/qrecc/train_original.json'
with open(train_file, 'r') as f:
    train_data = json.load(f)
print(len(train_data))

paraphrase_file = './data/qrecc/paraphrase.json'
paraphrase_data = json.load(open(paraphrase_file, 'r'))

swap_file = './data/qrecc/swap.json'
swap_data = json.load(open(swap_file, 'r'))
print(len(swap_data))

shift_file = './data/qrecc/shift.json'
shift_data = json.load(open(shift_file, 'r'))
print(len(shift_data))

extend_file = './data/qrecc/extend.json'
extend_data = json.load(open(extend_file, 'r'))
print(len(extend_data))

dependency_file = f"./data/qrecc/dependency.json"
dependency_data = json.load(open(dependency_file, 'r'))
print(len(dependency_data))

rnd = random.Random(0)
def token_deletion(sent, ratio=0.5):
    tokens = sent.split()
    num_to_delete = int(round(len(tokens) * ratio))
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if len(cand_indexes) >= 1 and token.startswith("##"):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    rnd.shuffle(cand_indexes)
    output_tokens = list(tokens)
    deleted_terms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(deleted_terms) >= num_to_delete:
            break
        if len(deleted_terms) + len(index_set) > num_to_delete:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = "[token_mask]"
            output_tokens[index] = masked_token
            deleted_terms.append((index, tokens[index]))
    assert len(sent.split()) == len(output_tokens)
    assert len(deleted_terms) <= num_to_delete
    return " ".join(output_tokens)

def augmentation(sample_id, history, question, strategy):
    random_positions = -1
    if strategy == "turn_deletion_depend":
        if len(history)!=0:
            turn_del_pool = []
            for turn_id in range(len(history)//2):
                if sample_id not in dependency_data or turn_id+1 not in dependency_data[sample_id]:
                    turn_del_pool.append(turn_id)
            random_num = min(int(len(history)//2 * 0.5), len(turn_del_pool))
            random_positions = rnd.sample(turn_del_pool, random_num)
            for random_position in random_positions: 
                history[random_position*2] = "[turn_mask]"
                history[random_position*2+1] = "[turn_mask]"
        aug_sequence = history
        aug_question = question
    elif strategy == "token_deletion":
        aug_sequence = []
        for sent in history:
            sent_aug = token_deletion(sent)
            sent_aug += " "
            sent_aug = re.sub(r'(\[token_mask\] ){2,}', "[token_mask] ", sent_aug)
            sent_aug = sent_aug[:-1]
            aug_sequence.append(sent_aug)
        aug_question = token_deletion(question)
        aug_question += " "
        aug_question = re.sub(r'(\[token_mask\] ){2,}', "[token_mask] ", aug_question)
        aug_question = aug_question[:-1]
    elif strategy == "reorder_depend":
        if len(history)<=2:
            return history + [question]
        conv_id, turn_id = sample_id.split("_")
        dependency_list = [[]]
        for i in range(2, int(turn_id)):
            sample_id_tmp = conv_id+"_"+str(i)
            if sample_id_tmp not in dependency_data:
                dependency_list.append([])
                continue
            dependency_list.append(dependency_data[sample_id_tmp])
        swappable_turns = identify_swappable_turns(dependency_list)
        if len(swappable_turns)==0:
            return history + [question]
        change_pos = rnd.sample(swappable_turns, 1)[0]
        aug_sequence = history.copy()
        tmp = history[change_pos[1] * 2:change_pos[1] * 2 + 2]
        aug_sequence[change_pos[1] * 2:change_pos[1] * 2 + 2] = history[change_pos[0] * 2:change_pos[0] * 2 + 2]
        aug_sequence[change_pos[0] * 2:change_pos[0] * 2 + 2] = tmp
        aug_question = question
        return aug_sequence + [aug_question]
    else:
        assert False
    return aug_sequence + [aug_question]

for conv in tqdm(train_data):
    history = []
    for ind, turn in enumerate(conv['turns']):
        if len(turn["pos_doc_ids"]) == 0:
            history.append(turn['question'])
            history.append(turn['response'])
            continue
        
        sample_id = "{}_{}".format(conv['conv_id'], turn['turn_id'])
        expect_len = len(history)+1
        
        question = turn['question']
        response = turn['response']
        
        # paraphrase
        
        aug_ctx = {}
        aug_ctx['paraphrase'] = paraphrase_data[sample_id][:-1] + [question]
        
        # extend
        
        if ind == 0:
            pass
        else:
            query_cnt = len(history) // 2
            insert_ind = rnd.choice(range(query_cnt+1)) * 2
            if insert_ind<len(history):
                aug_ctx['extend'] = history[:insert_ind] + extend_data[sample_id] + history[insert_ind:] + [question]
            else:
                aug_ctx['extend'] = history + extend_data[sample_id] + [question]
        
        # swap
        
        aug_ctx['swap'] = swap_data[sample_id]
        
        # shift
        
        aug_ctx['shift'] = shift_data[sample_id]
            
        # rewrite
        history1 = history[:]
        history2 = history[:]
        history3 = history[:]
        question1 = question[:]
        question2 = question[:]
        question3 = question[:]
        aug_ctx['token_deletion'] = augmentation(sample_id, history1, question1, 'token_deletion')
        aug_ctx['turn_deletion_depend'] = augmentation(sample_id, history2, question2, 'turn_deletion_depend')
        aug_ctx['reorder_depend'] = augmentation(sample_id, history3, question3, 'reorder_depend')
        
        history.append(question)
        history.append(response)
        
        turn['aug_ctx'] = aug_ctx


aug_train_file = './data/qrecc/train_augmented.json'
with open(aug_train_file, "w") as output_json_file:
    json.dump(train_data, output_json_file, indent=4)
