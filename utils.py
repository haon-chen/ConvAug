from IPython import embed

import os
import json
import shutil
import pickle
import random
import numpy as np
from os.path import join as oj

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import AdamW

# from multiprocessing import Process

def check_dir_exist_or_build(dir_list, force_emptying:bool = False):
    for x in dir_list:
        if not os.path.exists(x):
            os.makedirs(x)
        elif len(os.listdir(x)) > 0:    # not empty
            if force_emptying:
                print("Forcing to erase all contens of {}".format(x))
                shutil.rmtree(x)
                os.makedirs(x)
            else:
                raise FileExistsError
        else:
            continue

def json_dumps_arguments(output_path, args):   
    with open(output_path, "w") as f:
        params = vars(args)
        if "device" in params:
            params["device"] = str(params["device"])
        f.write(json.dumps(params, indent=4))


def split_and_padding_neighbor(batch_tensor, batch_len):
    batch_len = batch_len.tolist()
    pad_len = max(batch_len)
    device = batch_tensor.device
    tensor_dim = batch_tensor.size(1)

    batch_tensor = torch.split(batch_tensor, batch_len, dim = 0)
    
    padded_res = []
    for i in range(len(batch_tensor)):
        cur_len = batch_tensor[i].size(0)
        if cur_len < pad_len:
            padded_res.append(torch.cat([batch_tensor[i], 
                                        torch.zeros((pad_len - cur_len, tensor_dim)).to(device)], dim = 0))
        else:
            padded_res.append(batch_tensor[i])

    padded_res = torch.cat(padded_res, dim = 0).view(len(batch_tensor), pad_len, tensor_dim)
    
    return padded_res

def get_has_gold_label_test_qid_set(qrel_file):
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    qids = set()
    for line in qrel_data:
        line = line.strip().split("\t")
        query = line[0]
        qids.add(query)
    return qids


def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path, high_protocol = False):
    with open(path, 'wb') as f:
        if high_protocol:  
            pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))



def load_collection(collection_file):   
    all_docs = {}
    with open(collection_file, "r") as f:
        for line in f:
            line = line.strip()
            try:
                line_arr = line.split("\t")
                pid = int(line_arr[0])
                passage = line_arr[1].rstrip()
                all_docs[pid] = passage
            except IndexError:
                print("bad passage")
            except ValueError:
                print("bad pid")
    return all_docs

def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)



def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer



class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                # print("yielding record")
                # print(rec)
                yield rec

class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(
                self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".
                format(key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number

def get_finished_sample_ids(output_file_path):
    finished_samples = {}
    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as f:
            data = f.readlines()
        for line in data:
            line = json.loads(line)
            finished_samples[line['sample_id']] = []
                
    return finished_samples    