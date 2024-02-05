from typing import Any, List, Optional, Literal
import copy
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HF_Dataset
from IPython import embed
from dataclasses import dataclass
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import random
import json
from tqdm import tqdm
import random

@dataclass
class ConvSearchSample:
    sample_idx: str
    raw_utt: str
    manual_utt: str
    history: List[str]
    pos_docs: List[str]
    neg_docs: List[str]

@dataclass
class ConvSearchSampleAug:
    sample_idx: str
    raw_utt: str
    manual_utt: str
    history: List[str]
    pos_docs: List[str]
    neg_docs: List[str]
    aug_ctx: object


class ConvSearchDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 use_data_percent: float,
                 only_last_response: bool,
                 negative_type = 'in_batch_neg',
                 neg_ratio = 3):
        
        self.negative_type = negative_type
        self.neg_ratio = neg_ratio
        self.data = self.read_data(data_path, use_data_percent, only_last_response)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
    def read_data(self, data_path: str, use_data_percent: float, only_last_response: bool):
        samples = []
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        if use_data_percent < 1.0:
            n = int(use_data_percent * len(data))
            data = random.sample(data, n)

        for conv in tqdm(data, desc="Processing {} data file...".format(data_path)):
            history = []
            for turn in conv['turns']:
                sample_idx = "{}_{}".format(conv['conv_id'], turn['turn_id'])
                raw_utt = turn['question']
                manual_utt = ""
                if "manual_rewrite" in turn:
                    manual_utt = turn['manual_rewrite']
                pos_docs, neg_docs = [], []
                if "pos_doc_text" in turn:
                    pos_docs = turn['pos_doc_text']
                if self.negative_type == "random_neg":
                    assert "random_neg_docs_text" in turn
                    random_neg_docs = []
                    for doc in turn['random_neg_docs_text']:
                        if len(random_neg_docs) == self.neg_ratio:
                            break
                        random_neg_docs.append(doc)
                    neg_docs = random_neg_docs
                elif self.negative_type == "bm25_hard_neg":
                    assert "bm25_hard_neg_docs_text" in turn
                    bm25hard_neg_docs = []
                    for doc in turn['bm25_hard_neg_docs_text']:
                        if len(bm25hard_neg_docs) == self.neg_ratio:
                            break
                        bm25hard_neg_docs.append(doc)
                    neg_docs = bm25hard_neg_docs
                elif self.negative_type == "in_batch_neg":
                    neg_docs = []
                else:
                    raise KeyError("Negative type: {} not implmeneted".format(self.negative_type))
                sample = ConvSearchSample(sample_idx, raw_utt, manual_utt, copy.deepcopy(history), pos_docs, neg_docs)
                samples.append(sample)

                # update history
                if only_last_response and "response" in turn and len(history) > 0:
                    history.pop()   # pop the last response
                history.append(raw_utt)
                if "response" in turn: 
                    history.append(turn['response'])
        
        return samples

    def filter_no_pos_sample(self):
        filtered_data = []
        for sample in self.data:
            if len(sample.pos_docs) > 0:
                filtered_data.append(sample)
        self.data = filtered_data
        logger.info("Filtered {} samples with no positive documents.".format(len(self.data)))            
    
    @staticmethod
    def form_session_reverse_concat(history, raw_utt, tokenizer, max_seq_length):
        to_concat = []
        to_concat.extend(history)
        to_concat.append(raw_utt)
        to_concat.reverse()
        sep_token = tokenizer.sep_token
        if sep_token is None:
           sep_token = tokenizer.eos_token
        text = " {} ".format(sep_token).join(to_concat)
        
        return text
    
    @staticmethod
    def train_cdr_collate_fn(batch: list, tokenizer, max_seq_length):
        sample_ids = []
        query_inputs = []
        pos_doc_inputs = []
        neg_doc_inputs = []
        for sample in batch:
            sample_ids.append(sample.sample_idx)
            text = ConvSearchDataset.form_session_reverse_concat(sample.history, sample.raw_utt, tokenizer, max_seq_length)
            query_inputs.append(text)
            pos_doc_inputs += sample.pos_docs
            neg_doc_inputs += sample.neg_docs
        
        query_input_encodings = tokenizer(query_inputs, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt')  
        pos_doc_input_encodings = tokenizer(pos_doc_inputs, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt')      
        neg_doc_input_encodings = None
        if len(neg_doc_inputs) > 0:
            neg_doc_input_encodings = tokenizer(neg_doc_inputs, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt') 
              
        return {'sample_ids': sample_ids,
                'query_input_encodings': query_input_encodings,
                'pos_doc_input_encodings': pos_doc_input_encodings,
                'neg_doc_input_encodings': neg_doc_input_encodings,}
        
            
    @staticmethod
    def test_dr_collate_fn(batch: list, tokenizer, max_seq_length, input_type):
        sample_ids = []
        input_texts = []
        for sample in batch:
            sample_ids.append(sample.sample_idx)
            if input_type == 'manual':
                text = sample.manual_utt
            elif input_type == 'raw':
                text = sample.raw_utt
            elif input_type == 'session':
                text = ConvSearchDataset.form_session_reverse_concat(sample.history, sample.raw_utt, tokenizer, max_seq_length)

            input_texts.append(text)
        
        input_encodings = tokenizer(input_texts, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt')       
        
        return {'sample_ids': sample_ids,
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask']}


class ConvSearchDatasetAug(Dataset):
    def __init__(self,
                 data_path: str,
                 use_data_percent: float,
                 only_last_response: bool,
                 negative_type = 'in_batch_neg',
                 neg_ratio = 3):
        
        self.negative_type = negative_type
        self.neg_ratio = neg_ratio
        
        self.data = self.read_data(data_path, use_data_percent, only_last_response)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
    def read_data(self, data_path: str, use_data_percent: float, only_last_response: bool):
        samples = []
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        if use_data_percent < 1.0:
            n = int(use_data_percent * len(data))
            data = random.sample(data, n)

        for conv in tqdm(data, desc="Processing {} data file...".format(data_path)):
            history = []
            for turn in conv['turns']:
                sample_idx = "{}_{}".format(conv['conv_id'], turn['turn_id'])
                raw_utt = turn['question']
                manual_utt = ""
                if "manual_rewrite" in turn:
                    manual_utt = turn['manual_rewrite']
                pos_docs, neg_docs = [], []
                aug_ctx = {}
                if "pos_doc_text" in turn:
                    pos_docs = turn['pos_doc_text']
                if "neg_doc_text" in turn:
                    neg_docs = turn['neg_doc_text']
                if "aug_ctx" in turn:
                    aug_ctx = turn['aug_ctx']
                if self.negative_type == "random_neg":
                    assert "random_neg_docs_text" in turn
                    random_neg_docs = []
                    for doc in turn['random_neg_docs_text']:
                        random_neg_docs.append(doc)
                        if len(random_neg_docs) == self.neg_ratio:
                            break
                    neg_docs = random_neg_docs
                elif self.negative_type == "bm25_hard_neg":
                    assert "bm25_hard_neg_docs_text" in turn
                    bm25hard_neg_docs = []
                    for doc in turn['bm25_hard_neg_docs_text']:
                        if len(bm25hard_neg_docs) == self.neg_ratio:
                            break
                        bm25hard_neg_docs.append(doc)
                    neg_docs = bm25hard_neg_docs
                elif self.negative_type == "in_batch_neg":
                    neg_docs = []
                else:
                    raise KeyError("Negative type: {} not implmeneted".format(self.negative_type))
                sample = ConvSearchSampleAug(sample_idx, raw_utt, manual_utt, copy.deepcopy(history), pos_docs, neg_docs, aug_ctx)
                samples.append(sample)

                # update history
                if only_last_response and "response" in turn and len(history) > 0:
                    history.pop()   # pop the last response
                history.append(raw_utt)
                if "response" in turn: 
                    history.append(turn['response'])
        
        return samples

    def filter_no_pos_sample(self):
        filtered_data = []
        for sample in self.data:
            if len(sample.pos_docs) > 0:
                filtered_data.append(sample)
        self.data = filtered_data
        logger.info("Filtered {} samples with no positive documents.".format(len(self.data)))
    
    @staticmethod
    def form_session_reverse_concat(history, raw_utt, tokenizer, max_seq_length):
        to_concat = []
        to_concat.extend(history)
        to_concat.append(raw_utt)
        to_concat.reverse()
        sep_token = tokenizer.sep_token
        if sep_token is None:
           sep_token = tokenizer.eos_token
        text = " {} ".format(sep_token).join(to_concat)
        
        return text
    
    @staticmethod
    def train_aug_cdr_collate_fn(batch: list, tokenizer, max_seq_length, aug_strategies, neg_strategies=[], neg_aug_ratio=1, pos_aug_assign=None, neg_aug_assign=None):
        sample_ids = []
        query_inputs = []
        oracle_inputs = []
        pos_doc_inputs = []
        neg_doc_inputs = []
        aug_query_inputs1 = []
        aug_query_inputs2 = []
        neg_aug_query_inputs = []
        need_replace = False
        for sample in batch:
            sample_ids.append(sample.sample_idx)
            text = ConvSearchDatasetAug.form_session_reverse_concat(sample.history, sample.raw_utt, tokenizer, max_seq_length)
            query_inputs.append(text)
            oracle_inputs.append(sample.manual_utt)
            pos_doc_inputs += sample.pos_docs
            neg_doc_inputs += sample.neg_docs
            if pos_aug_assign:
                strategy1, strategy2 = pos_aug_assign[sample.sample_idx]
            else:
                if len(aug_strategies)>0:
                    aug_strategy = []
                    for aug in aug_strategies:
                        if aug in sample.aug_ctx and len(sample.aug_ctx[aug]) != 0:
                            aug_strategy.append(aug)
                    
                    no_strategies = []
                    if int(sample.sample_idx.split('_')[1]) <= 2:
                        no_strategies.append('reorder')
                        no_strategies.append('reorder_depend')
                    if 'turn_deletion_depend' not in sample.aug_ctx and 'turn_deletion' not in aug_strategy:
                        aug_strategy.append('turn_deletion')
                    for strategy in no_strategies:
                        if strategy in aug_strategy:
                            aug_strategy.remove(strategy)
                            
                    strategy1 = random.choice(aug_strategy)
                    
                    strategy2 = random.choice(aug_strategy)
                    while strategy1 == strategy2:
                        strategy2 = random.choice(aug_strategy)
            if strategy1 == "" and strategy2 == "":
                aug_text1 = copy.deepcopy(text)
                aug_query_inputs1.append(aug_text1)
                aug_text2 = copy.deepcopy(text)
                aug_query_inputs2.append(aug_text2)
            elif len(aug_strategies)>0:    
                aug_text1 = ConvSearchDatasetAug.form_session_reverse_concat(sample.aug_ctx[strategy1][:-1], sample.aug_ctx[strategy1][-1], tokenizer, max_seq_length)
                aug_query_inputs1.append(aug_text1)
                aug_text2 = ConvSearchDatasetAug.form_session_reverse_concat(sample.aug_ctx[strategy2][:-1], sample.aug_ctx[strategy2][-1], tokenizer, max_seq_length)
                aug_query_inputs2.append(aug_text2)

            if neg_aug_assign:
                sampled_neg_strategies = neg_aug_assign[sample.sample_idx]
            else:
                neg_strategy = []
                for aug in neg_strategies:
                    if aug in sample.aug_ctx and len(sample.aug_ctx[aug]) != 0:
                        neg_strategy.append(aug)
                
                if len(neg_strategy)==0 and neg_aug_ratio!=0:
                    neg_aug_query_inputs.append("empty")
                    need_replace = True
                    continue
                if len(neg_strategy)<neg_aug_ratio:
                    neg_strategy = neg_strategy+[neg_strategy[0]]*(neg_aug_ratio-len(neg_strategy))
                sampled_neg_strategies = random.sample(neg_strategy, neg_aug_ratio)
            if len(sampled_neg_strategies)==0 and neg_aug_ratio!=0:
                neg_aug_query_inputs.extend(["empty"]*neg_aug_ratio)
                need_replace = True
                continue
            if len(sampled_neg_strategies)<neg_aug_ratio:
                sampled_neg_strategies = sampled_neg_strategies+[sampled_neg_strategies[0]]*(neg_aug_ratio-len(sampled_neg_strategies))
            for aug in sampled_neg_strategies:
                neg_aug_ctx = sample.aug_ctx[aug]
                neg_aug_text = ConvSearchDatasetAug.form_session_reverse_concat(neg_aug_ctx[:-1], neg_aug_ctx[-1], tokenizer, max_seq_length)
                neg_aug_query_inputs.append(neg_aug_text)
        replace = None
        empty_index = []
        if need_replace:
            for ind, neg in enumerate(neg_aug_query_inputs):
                if neg != "empty":
                    replace = neg
                else:
                    empty_index.append(ind)
            for ind in empty_index:
                neg_aug_query_inputs[ind] = replace
        
        query_input_encodings = tokenizer(query_inputs, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt') 
        oracle_input_encodings = tokenizer(oracle_inputs, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt') 
        
        pos_doc_input_encodings = tokenizer(pos_doc_inputs, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt')      
        neg_doc_input_encodings = None
        if len(neg_doc_inputs) > 0:
            neg_doc_input_encodings = tokenizer(neg_doc_inputs, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt') 
        
        aug_input_encodings1 = None
        if len(aug_query_inputs1) > 0:  
            aug_input_encodings1 = tokenizer(aug_query_inputs1, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt')  
        aug_input_encodings2 = None
        if len(aug_query_inputs2) > 0: 
            aug_input_encodings2 = tokenizer(aug_query_inputs2, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt')
        
        neg_aug_input_encodings = None
        if len(neg_aug_query_inputs) > 0:
            neg_aug_input_encodings = tokenizer(neg_aug_query_inputs, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt')
        
        return {'sample_ids': sample_ids,
                'query_input_encodings': query_input_encodings,
                'aug_input_encodings1': aug_input_encodings1,
                'aug_input_encodings2': aug_input_encodings2,
                'pos_doc_input_encodings': pos_doc_input_encodings,
                'neg_doc_input_encodings': neg_doc_input_encodings,
                'neg_aug_input_encodings': neg_aug_input_encodings,
                "oracle_input_encodings": oracle_input_encodings}
            
    @staticmethod
    def test_dr_collate_fn(batch: list, tokenizer, max_seq_length, input_type):
        sample_ids = []
        input_texts = []
        for sample in batch:
            sample_ids.append(sample.sample_idx)
            if input_type == 'manual':
                text = sample.manual_utt
            elif input_type == 'raw':
                text = sample.raw_utt
            elif input_type == 'session':
                text = ConvSearchDataset.form_session_reverse_concat(sample.history, sample.raw_utt, tokenizer, max_seq_length)

            input_texts.append(text)
        
        input_encodings = tokenizer(input_texts, padding="longest", max_length=max_seq_length, truncation=True, return_tensors='pt')       
        
        return {'sample_ids': sample_ids,
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask']}
