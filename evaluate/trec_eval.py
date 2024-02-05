import os
import json
import pytrec_eval
import numpy as np
from pprint import pprint
from IPython import embed
import pandas as pd
import torch.multiprocessing as mp

def trec_eval(run_trec_file, qrel_trec_file, retrieval_output_path, rel_threshold):
    # process run trec file
    with open(run_trec_file, 'r' )as f:
        run_data = f.readlines()
    runs = {}
    for line in run_data:
        line = line.split(" ")
        sample_id = line[0]
        doc_id = line[2]
        score = float(line[4])
        if sample_id not in runs:
            runs[sample_id] = {}
        runs[sample_id][doc_id] = score

    # process qrel trec file
    with open(qrel_trec_file, 'r') as f:
        qrel_data = f.readlines()
    qrels = {}
    qrels_ndcg = {}
    for line in qrel_data:
        record = line.strip().split("\t")
        if len(record) == 1:
            record = line.strip().split(" ")
        query = record[0]
        doc_id = record[2]
        rel = int(record[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][doc_id] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][doc_id] = rel
 

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res_others = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res_others.values()]
    mrr_list = [v['recip_rank'] for v in res_others.values()]
    recall_5_list = [v['recall_5'] for v in res_others.values()]
    recall_10_list = [v['recall_10'] for v in res_others.values()]
    recall_20_list = [v['recall_20'] for v in res_others.values()]
    recall_100_list = [v['recall_100'] for v in res_others.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res_ndcg = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res_ndcg.values()]
    
    # embed()
    # input()

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list),
            "NDCG@3": np.average(ndcg_3_list), 
        }

    
    print("---------------------Evaluation results:---------------------")    
    pprint(res)
    if retrieval_output_path:
        with open(os.path.join(retrieval_output_path, "all_metric.res"), "w") as f:
            f.write(json.dumps(res, indent=4))
        with open(os.path.join(retrieval_output_path, "turn_level_others.res"), "w") as f:
            f.write(json.dumps(res_others, indent=4))
        with open(os.path.join(retrieval_output_path, "turn_level_ndcg.res"), "w") as f:
            f.write(json.dumps(res_ndcg, indent=4))
    
    return res



def agg_res_with_maxp(run_trec_file):
    res_file = os.path.join(run_trec_file)
    with open(run_trec_file, 'r' ) as f:
        run_data = f.readlines()
    
    agg_run = {}
    for line in run_data:
        line = line.strip().split(" ")
        if len(line) == 1:
            line = line.strip().split('\t')
        sample_id = line[0]
        if sample_id not in agg_run:
            agg_run[sample_id] = {}
        doc_id = "_".join(line[2].split('_')[:2])
        score = float(line[4])

        if doc_id not in agg_run[sample_id]:
            agg_run[sample_id][doc_id] = 0
        agg_run[sample_id][doc_id] = max(agg_run[sample_id][doc_id], score)
    
    agg_run = {k: sorted(v.items(), key=lambda item: item[1], reverse=True) for k, v in agg_run.items()}
    with open(os.path.join(run_trec_file + ".agg"), "w") as f:
        for sample_id in agg_run:
            doc_scores = agg_run[sample_id]
            rank = 1
            for doc_id, real_score in doc_scores:
                rank_score = 2000 - rank
                f.write("{} Q0 {} {} {} {}\n".format(sample_id, doc_id, rank, rank_score, real_score, "ance"))
                rank += 1                
        
if __name__ == "__main__":
    pass