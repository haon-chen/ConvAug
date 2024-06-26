import os
import json
import argparse
import pytrec_eval
import numpy as np
from pprint import pprint
from IPython import embed

def eval_run_with_qrel(**eval_kwargs):
    rel_threshold = eval_kwargs["rel_threshold"] if "rel_threshold" in eval_kwargs else 1
    retrieval_output_path = eval_kwargs["retrieval_output_path"] if "retrieval_output_path" in eval_kwargs else "./"

    if "run" in eval_kwargs:
        runs = eval_kwargs["run"]
    else:
        assert "run_file" in eval_kwargs
        with open(eval_kwargs["run_file"], 'r' )as f:
            run_data = f.readlines()
        runs = {}
        for line in run_data:
            line = line.split(" ")
            sample_id = line[0]
            pid = line[2]
            rel = float(line[4])
            if sample_id not in runs:
                runs[sample_id] = {}
            runs[sample_id][pid] = rel

    assert "qrel_file" in eval_kwargs
    with open(eval_kwargs["qrel_file"], 'r') as f:
        qrel_data = f.readlines()

    qrels = {}
    qrels_ndcg = {}
    for line in qrel_data:
        line = line.strip().split(" ")
        qid = line[0]
        pid = line[2]
        rel = int(line[3])
        if qid not in qrels:
            qrels[qid] = {}
        if qid not in qrels_ndcg:
            qrels_ndcg[qid] = {}

        # for NDCG
        qrels_ndcg[qid][pid] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[qid][pid] = rel
 
    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

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
    eval_kwargs['res'] = res
    with open(os.path.join(retrieval_output_path, "res.txt"), "w") as f:
        f.write(json.dumps(eval_kwargs, indent=4))

    return res



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval results.")
    parser.add_argument("--run_file", type=str, help="Path to the run file.")
    parser.add_argument("--qrel_file", type=str, help="Path to the qrel file.")
    parser.add_argument("--rel_threshold", type=int, default=1, help="Relevance threshold.")
    parser.add_argument("--retrieval_output_path", type=str, default="./", help="Path to the output directory.")
    args = parser.parse_args()

    eval_kwargs = {
        "run_file": args.run_file,
        "qrel_file": args.qrel_file,
        "rel_threshold": args.rel_threshold,
        "retrieval_output_path": args.retrieval_output_path,
    }

    eval_run_with_qrel(**eval_kwargs)
