"""Evaluation script."""

import json
import collections
import numpy as np
import os
import pickle
from new_reader.new_data_structure import SeqQueryGraph_Info

        
def load_ground_truth(out_gt_dir, task_filter):
    gt_dict = collections.defaultdict(lambda: collections.defaultdict(set))
    tasks = []
    for task_gt in os.listdir(out_gt_dir):
        task = task_gt.split("_")[0]
        if ( task_filter[0] == "*" ) or ( task in task_filter ):
            tasks.append(task)
    for task in tasks:
        with open(os.path.join(out_gt_dir, task+"_gt.pkl"), mode="rb") as f:
            gt_dict[task] = pickle.load(f)
    return gt_dict

def generate_ground_truth(all_query, out_gt_dir):
    gt_dict = collections.defaultdict(lambda: collections.defaultdict(set))
    
    for run_type in all_query:
        run_type_query = all_query[run_type]
        for task_datasets in run_type_query:
            task_query = task_datasets.query_features
            for query in task_query:
                query_type = query.query_type
                query_str = query.format_str()
                query_answer = query.get_query_answer()

                # # for debug
                # if len(gt_dict[query_type][query_str]) >= 3 and query_answer==24865:
                #     print(run_type)
                #     print(query_type)
                #     print(query_str)
                #     print(gt_dict[query_type][query_str])
                #     print(query_answer)

                gt_dict[query_type][query_str].add(query_answer)

    for query_type in gt_dict:
        with open(os.path.join(out_gt_dir, query_type+"_gt.pkl"), mode="wb") as f:
            pickle.dump(gt_dict[query_type], f)
    return gt_dict

def batch_evaluation(
    task_ranks, 
    batch_query,
    batch_eval_results, 
    # flatten_eval_query_features, 
    gt_dict
    ):
    """
    Perform batch evaluation.
    """
    query_types = batch_query.query_types
    query_strs = batch_query.query_strs
    all_answers = batch_query.answers.squeeze()
    answers_mask = batch_query.batch_answer_mask.squeeze()
    query_answers = all_answers[answers_mask>=0]
    for i, result in enumerate(batch_eval_results):
        query_answer = query_answers[i].item()
        query_type = query_types[i]
        query_str = query_strs[i]
        rm_idx = gt_dict[query_type][query_str]
        rm_idx = [x for x in rm_idx if x != query_answer]
        for x in rm_idx: result[x] = -np.Inf
        sortidx = np.argsort(result)[::-1]
        rank = np.where(sortidx == query_answer)[0][0] + 1
        task_ranks[query_type].append(rank)

    return task_ranks


def compute_metrics(task_ranks, eval_metrics_file):
    """
    Combine the ranks from batches into final metrics.
    """
    eval_metrics = collections.defaultdict(lambda: collections.defaultdict(np.float))
    
    for task in task_ranks:
        single_task_ranks = np.array(task_ranks[task])
        eval_metrics[task]["mrr"] = np.mean(1.0 / single_task_ranks)
        eval_metrics[task]["hits1"] = np.mean(single_task_ranks <= 1.0)
        eval_metrics[task]["hits3"] = np.mean(single_task_ranks <= 3.0)
        eval_metrics[task]["hits5"] = np.mean(single_task_ranks <= 5.0)
        eval_metrics[task]["hits10"] = np.mean(single_task_ranks <= 10.0)

    with open(eval_metrics_file, "w") as fw:
        fw.write(json.dumps(eval_metrics, indent=4) + "\n")

    return eval_metrics
