from IPython import embed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
import os
sys.path.append('..')
sys.path.append('.')

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator
from tqdm import tqdm, trange
from transformers import (HfArgumentParser, 
                          TrainingArguments, 
                          BitsAndBytesConfig,
                          set_seed,
                          AutoTokenizer,
                          Trainer)
from os.path import join as oj
from utils import check_dir_exist_or_build, json_dumps_arguments, set_seed
from data_lib import ConvSearchDataset, ConvSearchDatasetAug
from models import load_model
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import pickle
import itertools
import math
# import wandb

def categorize_samples(difficulties, n_groups, reverse=False):
    sorted_items = sorted(difficulties.items(), key=lambda x: x[1], reverse=reverse)
    samples_per_group = len(sorted_items) // n_groups
    remainder = len(sorted_items) % n_groups
    groups = {f'Group {i+1}': [] for i in range(n_groups)}
    sample_group_indices = {}
    start_index = 0
    for i in range(n_groups):
        group_size = samples_per_group + (1 if i < remainder else 0)
        for item in sorted_items[start_index:start_index + group_size]:
            group_name = f'Group {i + 1}'
            groups[group_name].append(item[0])
            sample_group_indices[item[0]] = i + 1
        start_index += group_size

    return groups, sample_group_indices

def reorganize_and_average_with_sorting(dictionary):
    sum_dict = {}
    count_dict = {}
    for key, value in dictionary.items():
        _, b = key.rsplit('+', 1)
        sum_dict[b] = sum_dict.get(b, 0) + value
        count_dict[b] = count_dict.get(b, 0) + 1
    avg_dict = {b: sum_dict[b] / count_dict[b] for b in sum_dict}
    is_ascending = all(x <= y for x, y in zip(list(dictionary.values()), list(dictionary.values())[1:]))
    sorted_avg_dict = dict(sorted(avg_dict.items(), key=lambda x: x[1], reverse=not is_ascending))

    return sorted_avg_dict

def filter_dict(D, L):
    keys_to_remove = [key for key in D.keys() if key not in L]
    for key in keys_to_remove:
        del D[key]
        
def remove_pairs(run_pairs, no_pairs):
    new_pairs = [pair for pair in run_pairs if pair not in no_pairs and pair[::-1] not in no_pairs]
    return new_pairs

def filter_pos_dict_pair(D, L):
    keys_to_remove = [key for key in D.keys() if key.split("+")[0] not in L or key.split("+")[1] not in L]
    for key in keys_to_remove:
        del D[key]

def filter_neg_dict_pair(D, L):
    keys_to_remove = [key for key in D.keys() if key.split("+")[1] not in L]
    for key in keys_to_remove:
        del D[key]

def filter_neg_dict_with_pos_list_pair(D, L):
    keys_to_remove = [key for key in D.keys() if key.split("+")[0] not in L]
    for key in keys_to_remove:
        del D[key]
        
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.detokeninistic = True
    torch.backends.cudnn.benchmark = True

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs, temperature):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    score_mat = pos_scores.clone()
    if neg_doc_embs is not None:
        neg_ratio = int(neg_doc_embs.shape[0] / query_embs.shape[0])
        neg_scores = torch.sum(query_embs.unsqueeze(1) * neg_doc_embs.view(batch_size, neg_ratio, -1), dim = -1) # B * neg_ratio
        score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + neg_ratio)  in_batch negatives + neg_ratio other negatives

    label_mat = torch.arange(batch_size).to(query_embs.device)
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)

    return loss

# Define and parse arguments.
@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "the model name"})
    dataset_path: str = field(metadata={"help": "the dataset name"})
    negative_ratio: int = field(metadata={"help": "the neg ratio"})
    neg_aug_ratio: int = field(metadata={"help": "the neg aug ratio"})
    early_stop_epoch: float = field(metadata={"help": "the early stop epoch"})
    
    only_last_response: bool = field(metadata={"help": "Wether to only include the last respone or not."})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    use_data_percent: Optional[float] = field(default=1.0, metadata={"help": "The percent of training data to use."})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "The temperature of the ranking loss."})
    aug_temperature: Optional[float] = field(default=1.0, metadata={"help": "The temperature of the ranking loss."})
    aug_weight: Optional[float] = field(default=1.0, metadata={"help": "The temperature of the ranking loss."})
    
    min_lr: float = field(default=0.0, metadata={"help": "The minimum learning rate in the cosine annealing scheduler."})
    
    aug_strategy: str = field(default="", metadata={"help": "the aug strategy"})
    neg_strategy: str = field(default="", metadata={"help": "the neg aug strategy"})
    sample_method: str = field(default="random", metadata={"help": "the method to sample strategies"})
    aug_sim_file: str = field(default="", metadata={"help": "aug sim file"})
    difficulty_file: str = field(default="", metadata={"help": "difficulty file"})
    
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    
    negative_type: str = field(default='in_batch_neg', metadata={"help": "the negative type"})
    loss_fig_path: str = field(default='./loss_figures/aug/', metadata={"help": "the negative type"})


def main():
    set_seed(42)
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    print(script_args.aug_temperature)
    # 1. load model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # This means: fit the entire model on the GPU:0
        device_map = {"": dist.get_rank()}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    query_model = load_model(script_args.model_path, 
                            quantization_config=quantization_config,
                            device_map=device_map,
                            trust_remote_code=script_args.trust_remote_code,
                            torch_dtype=torch_dtype)

    doc_model = load_model(script_args.model_path, 
                           quantization_config=quantization_config,
                           device_map=device_map,
                           trust_remote_code=script_args.trust_remote_code,
                           torch_dtype=torch_dtype)
    
    doc_model.requires_grad_(False)
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    
    additional_tokens = 2
    tokenizer.add_tokens("[token_mask]")
    tokenizer.add_tokens("[turn_mask]")
    query_model.resize_token_embeddings(query_model.config.vocab_size + additional_tokens)
    
    aug_strategy = script_args.aug_strategy.split(",")
    if script_args.aug_strategy == "":
        aug_strategy = []
    neg_strategy = script_args.neg_strategy.split(",")
    
    if len(aug_strategy) == 0 and script_args.neg_aug_ratio == 0:
        script_args.sample_method == 'random'
    if len(aug_strategy) == 2 and len(neg_strategy) == 1:
        script_args.sample_method == 'random'
        
    pos_aug_assign = None
    neg_aug_assign = None
    
    if script_args.sample_method == 'difficulty':
        pos_aug_assign = {}
        neg_aug_assign = {}
        
        with open(script_args.aug_sim_file, 'rb') as file:
            aug_sim= pickle.load(file)
        with open(script_args.difficulty_file, 'rb') as file:
            difficulty = pickle.load(file)

        no_pairs = [('turn_deletion', 'turn_deletion_depend'), ('reorder', 'reorder_depend'), ('turn_deletion', 'reorder_depend'), ('turn_deletion_depend', 'reorder'), ('swap', 'shift')]
        pos_run_pairs = list(itertools.combinations(aug_strategy, 2))
        pos_run_pairs = remove_pairs(pos_run_pairs, no_pairs)
        neg_run_pairs = list(itertools.product(aug_strategy, neg_strategy))
        neg_run_pairs = remove_pairs(neg_run_pairs, no_pairs)

        hard2hard = True

        _, pos_groups = categorize_samples(difficulty, len(pos_run_pairs), reverse=hard2hard)
        pos_group_num = len(pos_run_pairs)
        _, neg_groups = categorize_samples(difficulty, len(neg_run_pairs), reverse=not hard2hard)
        neg_group_num = len(neg_run_pairs)

        for sample_id, posneg_dict in aug_sim.items():
            pos_aug, neg_aug = posneg_dict['pos_aug'], posneg_dict['neg_aug']
            
            tmp_aug_strategy = []
            no_strategies = []
            if int(sample_id.split('_')[1]) <= 2:
                no_strategies.append('reorder')
                no_strategies.append('reorder_depend')
            for strategy in aug_strategy:
                if strategy not in no_strategies:
                    tmp_aug_strategy.append(strategy)
                            
            filter_pos_dict_pair(pos_aug, tmp_aug_strategy)
            filter_neg_dict_pair(neg_aug, neg_strategy)
            pos_aug_list, neg_aug_list = list(pos_aug.keys()), list(neg_aug.keys())
            pos_index = pos_groups[sample_id]
            neg_index = neg_groups[sample_id]
            
            pos_ratio = (pos_index-1) / float(pos_group_num)
            neg_ratio = (neg_index-1) / float(neg_group_num)
            if len(pos_aug_list)>0:
                pos_aug_assign[sample_id] = pos_aug_list[int(pos_ratio*len(pos_aug_list))].split("+")
            else:
                pos_aug_assign[sample_id] = ["", ""]
            
            filter_neg_dict_with_pos_list_pair(neg_aug, pos_aug_assign[sample_id])
            re_neg_aug = reorganize_and_average_with_sorting(neg_aug)
            neg_aug_list = list(re_neg_aug.keys())
            if script_args.neg_aug_ratio >= len(neg_aug_list):
                neg_aug_assign[sample_id] = neg_aug_list
            else:
                neg_aug_assign[sample_id] = neg_aug_list[int(neg_ratio*len(neg_aug_list)):int(neg_ratio*len(neg_aug_list))+script_args.neg_aug_ratio]
            
        
    # 2. build training dataset
    train_dataset = ConvSearchDatasetAug(script_args.dataset_path, 
                                      script_args.use_data_percent, 
                                      script_args.only_last_response,
                                      script_args.negative_type,
                                      script_args.negative_ratio)
    train_dataset.filter_no_pos_sample()
    
    # 3. train
    class CdrTrainer(Trainer):
        cl_losses = []
        ranking_losses = []
        def get_loss_plt(self):
            return self.cl_losses, self.ranking_losses
        def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
            if t is None:
                return None
            t = t.contiguous()
            
            cuda_device = f'cuda:{torch.distributed.get_rank()}'
            world_size = dist.get_world_size()
            local_size = torch.tensor(t.shape[0], device=cuda_device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)
            size_diff = max(all_sizes).item() - local_size.item()
            if size_diff > 0:
                padding = torch.zeros((size_diff, t.shape[1]), device=cuda_device, dtype=t.dtype)
                t = torch.cat((t, padding))

            all_tensors_padded = [torch.empty_like(t) for _ in range(world_size)]
            dist.all_gather(all_tensors_padded, t)
            # cut the padding
            all_tensors = []
            for iter_t, size in zip(all_tensors_padded, all_sizes):
                all_tensors.append(iter_t[:size])
            # always put tensors of the current rank at the first place
            all_tensors[dist.get_rank()] = t
            all_tensors.pop(dist.get_rank())
            all_tensors = [t] + all_tensors
        
            all_tensors = torch.cat(all_tensors, dim=0)

            return all_tensors
    
        def compute_loss(self, model, inputs):
            torch.autograd.set_detect_anomaly(True)
            sample_ids = inputs.pop('sample_ids')
            query_input_encodings = inputs.pop('query_input_encodings')
            pos_doc_input_encodings = inputs.pop('pos_doc_input_encodings')
            neg_doc_input_encodings = inputs.pop('neg_doc_input_encodings')
            
            aug_input_encodings1 = inputs.pop('aug_input_encodings1')
            aug_input_encodings2 = inputs.pop('aug_input_encodings2') # B, aug_ratio, 768
            
            neg_aug_input_encodings = inputs.pop('neg_aug_input_encodings')
            
            doc_model.to(pos_doc_input_encodings['input_ids'].device)
            
            query_embs = model(**query_input_encodings) # B,768
            pos_doc_embs = doc_model(**pos_doc_input_encodings).detach()
            
            neg_doc_embs = None
            if neg_doc_input_encodings:
                neg_doc_embs = doc_model(**neg_doc_input_encodings).detach()
            
            pos_doc_embs = self._dist_gather_tensor(pos_doc_embs)
            neg_doc_embs = self._dist_gather_tensor(neg_doc_embs)
            
            ranking_loss = cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs, script_args.temperature)
            
            if aug_input_encodings1!=None and aug_input_encodings2!=None:
                cl_loss, acc = model.module.compute_augCL_loss(query_embs, aug_input_encodings1, aug_input_encodings2, script_args.aug_temperature, neg_aug_input_encodings)
                cl_loss = cl_loss.mean()
                self.cl_losses.append(cl_loss.cpu().item())
                self.ranking_losses.append(ranking_loss.cpu().item())
                acc = acc.mean()
                loss = ranking_loss + cl_loss * script_args.aug_weight
            else:
                loss = ranking_loss
                self.ranking_losses.append(ranking_loss.cpu().item())
            
            return loss
    
    trainer = CdrTrainer(
        model=query_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=lambda x: train_dataset.train_aug_cdr_collate_fn(x, tokenizer, script_args.max_seq_length, aug_strategy, neg_strategy, script_args.neg_aug_ratio, pos_aug_assign, neg_aug_assign),
    )
    trainer.train()
    
    
if __name__ == '__main__':
    main()