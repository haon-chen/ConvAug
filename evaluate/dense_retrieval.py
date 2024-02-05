from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import json
import faiss
import time
import copy
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint
from os.path import join as oj

import torch
from torch.utils.data import DataLoader
from utils import check_dir_exist_or_build, json_dumps_arguments, set_seed
from data_lib import ConvSearchDataset
from transformers import AutoTokenizer, AutoModel
from models import load_model
from trec_eval import trec_eval, agg_res_with_maxp


'''
Test process, perform dense retrieval on collection (e.g., MS MARCO):
1. get args
2. establish index with Faiss on GPU for fast dense retrieval
3. load the model, build the test query dataset/dataloader, and get the query embeddings. 
4. iteratively searched on each passage block one by one to got the retrieved scores and passge ids for each query.
5. merge the results on all pasage blocks
6. output the result
'''



def build_faiss_index(n_gpu_for_faiss, embedding_size):
    logger.info("Building index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = n_gpu_for_faiss
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(embedding_size)  
    index = None
    if n_gpu_for_faiss > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index
        logger.warning("Use cpu for faiss!")

    return index

@torch.no_grad()
def get_query_embs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = load_model(args.model_path, 
                       quantization_config=None,
                       device_map=None,
                       trust_remote_code=None,
                       torch_dtype=None)
    
    model.eval()
    model.to(args.device)
    
    dataset = ConvSearchDataset(args.data_path, use_data_percent=args.use_data_percent, only_last_response=args.only_last_response)
    
    dataloader = DataLoader(dataset, 
                            batch_size=args.per_gpu_batch_size, 
                            shuffle=False, 
                            collate_fn=lambda x: dataset.test_dr_collate_fn(x, tokenizer, args.max_seq_length, args.input_type))

    # get all sample_ids that needed to be evaluated
    needed_to_be_eval = set()
    with open(args.qrel_path, 'r') as f:
        for line in f:            
            if '\t' in line:
                sid, _, _, _= line.strip('\n').split('\t')
            else:
                sid, _, _, _= line.strip('\n').split(' ')
            needed_to_be_eval.add(sid)
    
    query_embs = []
    qids = []
    for batch in tqdm(dataloader, desc="encoding queries..."):
        inputs = {k: v.to(args.device) for k, v in batch.items() if k not in {"sample_ids"}}
        embs = model(**inputs)
        embs = embs.detach().cpu().numpy()
        for i, qid in enumerate(batch['sample_ids']):
            if qid in needed_to_be_eval:
                qids.append(qid)
                query_embs.append(embs[i].reshape(1, -1))
   
    query_embs = np.concatenate(query_embs, axis=0)
    logger.info("#Total queries for evaluation: {}".format(len(query_embs)))
    
    return qids, query_embs


def search_one_by_one_with_faiss(dense_index_path, index, num_split_block, top_n, query_embs):
    merged_candidate_matrix = None
    block_id = 1
    for emb_block_name in os.listdir(dense_index_path):
        if "emb_block" not in emb_block_name:
            continue  
        logger.info("Loading block {}, ".format(block_id) + emb_block_name)
        passage_embedding = None
        passage_embedding2id = None
        embid_block_name = emb_block_name.replace("emb_block", "embid_block")

        try:
            with open(oj(dense_index_path, emb_block_name), 'rb') as handle:
                passage_embedding = pickle.load(handle)
            with open(oj(dense_index_path, embid_block_name), 'rb') as handle:
                passage_embedding2id = pickle.load(handle)
                if isinstance(passage_embedding2id, list):
                    passage_embedding2id = np.array(passage_embedding2id)
        except Exception as e:
            logger.error("An unexpected error occurred while loading block " + emb_block_name + ": " + str(e))

        logger.info('passage embedding shape: ' + str(passage_embedding.shape))
        logger.info("query embedding shape: " + str(query_embs.shape))
        
        passage_embeddings = np.array_split(passage_embedding, num_split_block)
        passage_embedding2ids = np.array_split(passage_embedding2id, num_split_block)
        for split_idx in range(len(passage_embeddings)):
            passage_embedding = passage_embeddings[split_idx]
            passage_embedding2id = passage_embedding2ids[split_idx]
            
            logger.info("Adding block {}: {}, split {} into index...".format(block_id, emb_block_name, split_idx))
            index.add(passage_embedding)
            
            # ann search
            tb = time.time()
            D, I = index.search(query_embs, top_n)
            elapse = time.time() - tb
            logger.info({
                'time cost': elapse,
                'query num': query_embs.shape[0],
                'time cost per query': elapse / query_embs.shape[0]
            })

            candidate_id_matrix = passage_embedding2id[I] # passage_idx -> passage_id
            D = D.tolist()
            candidate_id_matrix = candidate_id_matrix.tolist()
            candidate_matrix = []

            for score_list, passage_list in zip(D, candidate_id_matrix):
                candidate_matrix.append([])
                for score, passage in zip(score_list, passage_list):
                    candidate_matrix[-1].append((score, passage))
                assert len(candidate_matrix[-1]) == len(passage_list)
            assert len(candidate_matrix) == I.shape[0]

            index.reset()
            del passage_embedding
            del passage_embedding2id

            if merged_candidate_matrix == None:
                merged_candidate_matrix = candidate_matrix
                continue
            
            # merge
            merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
            merged_candidate_matrix = []
            for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                            candidate_matrix):
                p1, p2 = 0, 0
                merged_candidate_matrix.append([])
                while p1 < top_n and p2 < top_n:
                    if merged_list[p1][0] >= cur_list[p2][0]:
                        merged_candidate_matrix[-1].append(merged_list[p1])
                        p1 += 1
                    else:
                        merged_candidate_matrix[-1].append(cur_list[p2])
                        p2 += 1
                while p1 < top_n:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                while p2 < top_n:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1
                    
        block_id += 1
        
    merged_D, merged_I = [], []

    for merged_list in merged_candidate_matrix:
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)

    logger.info(merged_D.shape)
    logger.info(merged_I.shape)

    return merged_D, merged_I


def output_res(args,
               sample_ids,
               retrieved_scores_mat,
               retrieved_pid_mat):
    
    qids_to_ranked_candidate_passages = {}
    topN = args.top_n

    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()

        top_ann_pid = retrieved_pid_mat[query_idx].copy()
        top_ann_score = retrieved_scores_mat[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0

        tmp = [(0, 0)] * topN
        # tmp_ori = [0] * topN
        qids_to_ranked_candidate_passages[query_idx] = tmp

        for pred_pid, score in zip(selected_ann_idx, selected_ann_score):
            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_idx][rank] = (pred_pid, score)
                rank += 1
                seen_pid.add(pred_pid)
    # embed()
    # input()
    # write to file
    logger.info('begin to write the output...')

    output_file = oj(args.output_path, 'run.json')
    output_trec_file = oj(args.output_path, 'run.trec')
    
    collection = {}
    if args.collection_path != "":
        with open(args.collection_path, 'r') as f:
            for line in tqdm(f, desc='loading colletion...'):
                doc_id, doc = line.strip('\n').split('\t')[:2]
                collection[str(doc_id)] = doc
        
    with open(output_file, "w") as f, open(output_trec_file, "w") as g:
        for idx, passages in qids_to_ranked_candidate_passages.items():
            for i in range(topN):
                pid, score = passages[i]
                if len(collection) > 0:
                    passage = collection[str(pid)]
                else:
                    passage = ""
                f.write(
                        json.dumps({
                            "sample_id": str(sample_ids[idx]),
                            "doc": passage,
                            "pid": pid,
                            "rank": i,
                            "retrieval_score": score,
                        }) + "\n")
                
                g.write("{} Q0 {} {} {}".format(sample_ids[idx], pid, i+1, topN-i, "model"))
                g.write('\n')

    logger.info("output file write ok at {}".format(args.output_path))
    
    # evaluation
    if "cast21" in args.qrel_path:
        agg_res_with_maxp(output_trec_file)
        output_trec_file += ".agg"
    trec_eval(output_trec_file, args.qrel_path, args.output_path, args.rel_threshold)



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=64, help="Max seq length")
    parser.add_argument("--dense_index_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--qrel_path", type=str, required=True)
    parser.add_argument("--rel_threshold", type=int, required=True, help="CAsT-20: 2, Others: 1")
    parser.add_argument("--only_last_response", action="store_true", help="Whether to only include the last_response or not. if input_type == 'session'.")
    parser.add_argument("--input_type", choices=['manual', 'raw', 'session'], required=True)
    parser.add_argument("--per_gpu_batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_size", type=int, default=768, help="should be passage emb size")
    parser.add_argument("--n_gpu_for_faiss", type=int, default=1, help="should be set if use_gpu_in_faiss")
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--use_data_percent", type=float, default=1.0)
    parser.add_argument("--num_split_block", type=int, default=1, help="further split each block into several sub-blocks to reduce gpu memory use.")
    parser.add_argument("--collection_path", type=str, default="", help="the original collection path")
    
    # output file
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")
    

    # main
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    check_dir_exist_or_build([args.output_path], args.force_emptying_dir)
    json_dumps_arguments(oj(args.output_path, "parameters.txt"), args)
 
    logger.info("---------------------The arguments are:---------------------")
    pprint(args)
    
    return args



if __name__ == '__main__':
    args = get_args()
    set_seed(args) 
    
    index = build_faiss_index(args.n_gpu_for_faiss, args.embedding_size)
    sample_ids, query_embs = get_query_embs(args)    
    # pid_mat: corresponding passage ids
    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(args.dense_index_path, 
                                                                           index,
                                                                           args.num_split_block,
                                                                           args.top_n,
                                                                           query_embs) 
    
    output_res(args, sample_ids, retrieved_scores_mat, retrieved_pid_mat)
    
    logger.info("Retrieval finish!")