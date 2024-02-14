from tqdm import tqdm
from IPython import embed
import numpy as np
import json
import copy
import pickle
from sentence_transformers import SentenceTransformer as SBert, util
from bertopic import BERTopic

sentence_model = SBert('all-MiniLM-L6-v2')
model_dir = "MaartenGr/BERTopic_Wikipedia"
topic_model = BERTopic.load(model_dir)

def calculate_confidence(probs, exclude_indices=[]):
    probs = np.delete(probs, exclude_indices)
    sorted_probs = np.sort(probs)[::-1]
    return sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

def count_topics_with_confidence(data):
    topics_count = {}
    excluded_topics = {}
    for key in data.keys():
        conversation_id, turn_id = key.split('_')
        probs = data[key]
        
        if turn_id == '1':
            confidence = calculate_confidence(probs)
            topics_count[key] = (1, confidence) 
            excluded_topics[conversation_id] = [np.argmax(probs)]
        else:
            current_confidence = calculate_confidence(probs, exclude_indices=excluded_topics[conversation_id])

            if current_confidence >= confidence:
                new_topic_index = np.argmax(np.delete(probs, excluded_topics[conversation_id]))
                excluded_topics[conversation_id].append(new_topic_index)
                topics_count[key] = (topics_count[conversation_id + '_' + str(int(turn_id)-1)][0] + 1, current_confidence)
                confidence = current_confidence
            else:
                topics_count[key] = (topics_count[conversation_id + '_' + str(int(turn_id)-1)][0], current_confidence)
        
    return topics_count


train_file = '/data/train.json'
with open(train_file, 'r') as f:
    train_data = json.load(f)
print(len(train_data))
sid2conv = {}
for conv in tqdm(train_data):
    history = []
    for ind, turn in enumerate(conv['turns']):
        question = turn['question']
        response = turn['response']
        
        if len(turn["pos_doc_ids"]) == 0:
            history.append(question)
            history.append(response)
            continue
        
        sample_id = "{}_{}".format(conv['conv_id'], turn['turn_id'])
        sid2conv[sample_id] = copy.deepcopy(history) + [question, response]
        history.append(question)
        history.append(response)
print(len(sid2conv))

sid2convemb = {}
sids = list(sid2conv.keys())
convs = [" ".join(conv) for conv in list(sid2conv.values())]
conv_embeddings = sentence_model.encode(convs, show_progress_bar=True)
for sid, embedding in tqdm(zip(sids, conv_embeddings)):
    sid2convemb[sid] = embedding
    
sid2probs = {}
for sid, embedding in tqdm(sid2convemb.items()):
    doc = sid2conv[sid]
    conv_doc = " ".join(doc)
    embedding = embedding.reshape(1,embedding.shape[0])
    topics, probs = topic_model.transform([conv_doc], embedding)
    probs.sort()
    sid2probs[sid] = probs[0]

representative_topics_count = count_topics_with_confidence(sid2probs)

topic_cnts_file = '/data/topic_cnts.pkl'
with open(topic_cnts_file, 'wb') as file:
    pickle.dump(representative_topics_count, file)