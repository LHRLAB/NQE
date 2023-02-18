from __future__ import print_function
from __future__ import division
from gc import collect

import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
# import json
# import ujson as json
import orjson as json
import numpy as np
import collections
import logging
import time
import copy

import pickle

import os
import itertools
from typing import Sequence
from new_reader.vocab_reader import Vocabulary
from new_reader.new_data_structure import \
    HRF_Info, HRF_Features, Logic_Info, Logic_Features, \
        SeqQueryGraph_Info, SeqQueryGraph_Features, SeqQueryGraphBatch

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())


def generate_gt_dict(all_task_filter, vocabulary, run_type_ls, config):
    # gt_dict = collections.defaultdict(lambda: collections.defaultdict(set))

    input_dir = config["dataset_dir"]
    max_arity = config["max_arity"]
    max_seq_length = config["max_seq_len"]
    # tasks = []
    out_gt_dir=config["gt_dir"]

    tasks_ls = os.listdir(input_dir)
    tasks_ls.sort()
    if not os.path.exists(out_gt_dir):
        os.mkdir(out_gt_dir)
    else:
        logger.info("gt already exists.")
        return
    for task in tasks_ls:
        gt_dict = collections.defaultdict(set)
        for run_type in run_type_ls:
            # tasks_str = config[run_type+"_tasks"]

            if ( all_task_filter[0] != "*" ) and ( task not in all_task_filter ):
                continue
            logger.info("reading queries tasks: %s" % task)
            task_dir = os.path.join(input_dir, task)
            if os.path.isfile(task_dir):
                continue
            # tasks.append(task)
            input_file = os.path.join(task_dir, run_type+".json")
            gt_dict = generator_convert_info_to_features(
                    gt_dict=gt_dict,
                    input_file=input_file,
                    task=task,
                    vocabulary=vocabulary,
                    max_arity=max_arity,
                    max_seq_length=max_seq_length,
                    run_type=run_type,
                )

        with open(os.path.join(out_gt_dir, task+"_gt.pkl"), mode="wb") as f:
            pickle.dump(gt_dict, f)
    return       
        




        
def generator_convert_info_to_features(gt_dict, input_file, task, vocabulary, max_arity, max_seq_length, run_type):
    query_type = task

    total_hrf = 0
    arity_stats = np.zeros(shape=max_arity+1)
    
    logic_str = ["direct", "not", "and", "or"]
    type_id_map = {}
    for id, type_str in enumerate(logic_str):
        type_id_map[type_str] = id
    logic_stats = [0, 0, 0, 0]

    max_seq_length = 2 * ( max_arity - 2 ) + 3

    time_start = time.time()
    count = 0
    with open(input_file, 'r', encoding='utf-8', newline="\n") as fr:
        while True:
            line_data = fr.readline()
            if line_data:
                new_dict = json.loads(line_data)

                for query_idx in new_dict:

                    query_dict = new_dict[query_idx]

                    len_bigger = False

                    op_num = len(query_dict)
                    assert op_num % 2 == 0
                    hrf_num = op_num // 2
                    hrf_ls = []
                    logic_ls = []
                    for order in range(hrf_num):
                        hrf_id = "hrf" + str(order)
                        hrf_dict = query_dict[hrf_id]
                        seq_len = hrf_dict["N"] + 1

                        if seq_len > max_seq_length:
                            len_bigger = True
                            break

                        assert seq_len % 2 == 1
                        arity = ( seq_len + 1 ) // 2
                        hrf_info = HRF_Info(
                            N=hrf_dict["N"],
                            hrf=hrf_dict["hrf"],
                            out_pos=hrf_dict["out_pos"],
                            answers=hrf_dict["answers"],
                            in_pos=hrf_dict["in_pos"],
                            in_vars=hrf_dict["in_vars"],
                        )

                        arity_stats[arity] += 1
                        
                        logic_id = "logic" + str(order)     
                        logic_dict = query_dict[logic_id]       
                        logic_info = Logic_Info(
                            type=logic_dict["type"],
                            in_vars=logic_dict["in_vars"],
                        )

                        logic_stats[type_id_map[logic_dict["type"]]] += 1

                        hrf_ls.append(hrf_info)
                        logic_ls.append(logic_info)

                    if len_bigger:
                        break

                    hrf_features_ls = calc_hrf_features(hrf_ls, vocabulary, max_arity, max_seq_length)
                    logic_str = ["direct", "not", "and", "or"]
                    type_id_map = {}
                    for id, type_str in enumerate(logic_str):
                        type_id_map[type_str] = id
                    logic_features = calc_logic_features(logic_ls, type_id_map)
                    for hrf_features in hrf_features_ls:

                        count += 1
                        if count % 5000 == 0:
                            time_now = time.time()
                            time_consumption = time_now - time_start
                            time_sec = time_consumption % 60
                            time_min = ( time_consumption % 3600 ) // 60 
                            time_hour = time_consumption // 3600
                            logger.info("query_type: %s-%s read queries: %d time_consumption: %dh %dm %.2fs" % (query_type, run_type, count, time_hour, time_min, time_sec) )
                        
                        query = SeqQueryGraph_Features(
                            query_type=query_type,
                            num_hrf=hrf_num,
                            hrf_features=hrf_features,
                            logic_features=logic_features,
                        )
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

                        gt_dict[query_str].add(query_answer)

            else:
                break
    
    return gt_dict









def collate_batch_query(queries_features: Sequence[SeqQueryGraph_Features]):
    
    input_ids_ls = []
    input_mask_ls = []
    edge_orders_ls = []
    hrf_start_ls = []
    var_in_mask_ls = []
    # var_in_batch_mask_ls = []
    var_out_mask_ls = []
    # var_out_batch_mask_ls = []
    out_pos_ls = []
    answers_ls = []
    mask_type_ls = []
    logic_in_ids_ls = []
    # logic_in_batch_ids = []
    logic_types_ls = []
    batch_answer_mask_ls = []

    edge_offset = 0
    for local_id, query_features in enumerate(queries_features):
        query_type = query_features.query_type
        num_hrf = query_features.num_hrf 
        hrf_features = query_features.hrf_features
        logic_features = query_features.logic_features
        
        input_ids = hrf_features.input_ids
        input_mask = hrf_features.input_mask
        hrf_var_in_mask = hrf_features.var_in_mask
        answer = hrf_features.answer
        out_pos = hrf_features.out_pos 
        var_out_mask = hrf_features.var_out_mask
        mask_type = hrf_features.mask_type
        edge_order = hrf_features.edge_order

        logic_types = logic_features.type
        logic_in_vars = logic_features.in_vars

        input_ids_ls.append(input_ids)
        input_mask_ls.append(input_mask)
        edge_orders_ls.append(edge_order)
        hrf_start_ls.append(torch.full_like(input=edge_order, fill_value=edge_offset, dtype=torch.int64))
        var_in_mask_ls.append(hrf_var_in_mask)
        # var_in_batch_mask_ls = []
        var_out_mask_ls.append(var_out_mask)
        # var_out_batch_mask_ls = []
        out_pos_ls.append(out_pos)
        answers_ls.append(answer)
        mask_type_ls.append(mask_type)
        logic_in_ids_ls.append(logic_in_vars)
        # logic_in_batch_ids = []
        logic_types_ls.append(logic_types)
        hrf_answer_mask = - torch.ones_like(edge_order)
        hrf_answer_mask[0, -1] = 1
        batch_answer_mask_ls.append(hrf_answer_mask)     
        edge_offset += num_hrf
    input_ids_t = torch.cat(input_ids_ls, dim=-1)
    input_mask_t = torch.cat(input_mask_ls, dim=-1)
    edge_orders_t = torch.cat(edge_orders_ls, dim=-1)
    hrf_start_t = torch.cat(hrf_start_ls, dim=-1)
    var_in_mask_t = torch.cat(var_in_mask_ls, dim=-1)
    var_in_batch_mask = torch.div(var_in_mask_t, 2, rounding_mode='trunc') + hrf_start_t
    var_out_mask_t = torch.cat(var_out_mask_ls, dim=-1)
    var_out_batch_mask = var_out_mask_t.clone().detach()
    var_out_batch_mask_aux = torch.div(var_out_mask_t, 2, rounding_mode='trunc') + hrf_start_t 
    var_out_batch_mask[var_out_mask_t>=0] = var_out_batch_mask_aux[var_out_mask_t>=0]
    out_pos_t = torch.cat(out_pos_ls, dim=-1)
    answers_t = torch.cat(answers_ls, dim=-1)
    mask_type_t = torch.cat(mask_type_ls, dim=-1)
    logic_in_ids_t = torch.cat(logic_in_ids_ls, dim=-1)
    logic_in_batch_ids = torch.div(logic_in_ids_t, 2, rounding_mode='trunc') + hrf_start_t
    logic_types_t = torch.cat(logic_types_ls, dim=-1)
    batch_answer_mask_t = torch.cat(batch_answer_mask_ls, dim=-1)       
    return SeqQueryGraphBatch(
        input_ids=input_ids_t,
        input_mask=input_mask_t,
        edge_orders=edge_orders_t,
        hrf_start=hrf_start_t,
        var_in_mask=var_in_mask_t,
        var_in_batch_mask=var_in_batch_mask,
        var_out_mask=var_out_mask_t,
        var_out_batch_mask=var_out_batch_mask,
        out_pos=out_pos_t,
        answers=answers_t,
        mask_type=mask_type_t,
        logic_in_ids=logic_in_ids_t,
        logic_in_batch_ids=logic_in_batch_ids,
        logic_types=logic_types_t,
        batch_answer_mask=batch_answer_mask_t,
    )

def calc_hrf_features(hrf_infos, vocabulary, max_arity, max_seq_length):
    hrf_features = []
    hrf_num = len(hrf_infos)
    vec_size = [1, hrf_num]
    mat_size = [max_seq_length, hrf_num]
    N_t = torch.zeros(size=vec_size, dtype=torch.int64)
    input_ids = torch.zeros(size=mat_size, dtype=torch.int64)
    input_mask = torch.zeros(size=mat_size, dtype=torch.int64)
    var_in_mask = - torch.ones(size=mat_size, dtype=torch.int64)
    answer_t = - torch.ones(size=vec_size, dtype=torch.int64)
    out_pos_t = - torch.ones(size=vec_size, dtype=torch.int64)
    var_out_mask =  - torch.ones(size=mat_size, dtype=torch.int64)
    mask_type = torch.ones(size=vec_size, dtype=torch.int64)
    edge_order = torch.arange(hrf_num).unsqueeze(0)
    for order, hrf_info in enumerate(hrf_infos):
        N = hrf_info.N
        hrf = hrf_info.hrf
        out_pos = hrf_info.out_pos
        answers = hrf_info.answers
        in_pos = hrf_info.in_pos
        in_vars = hrf_info.in_vars

        N_t[0, order] = N
        input_ids[:N+1, order] = torch.tensor(vocabulary.convert_tokens_to_ids(VAR2MASK(hrf)))
        input_mask[:N+1, order] = 1
        len(in_pos) == len(in_vars)
        var_in_mask[in_pos, order] = torch.tensor(in_vars, dtype=torch.int64)
        out_pos_t[0, order] = out_pos
        var_out_mask[out_pos, order] = 2 * order 
        if out_pos % 2 == 1:
            mask_type[0, order] = -1

        if order < hrf_num - 1:
            continue
        
        # if len(answers["easy"]) >= 1:

        #     # for debug  
        #     count = 0
        #     last_str = ""
        #     last_answer_t = torch.zeros(1)

        for answer in answers["easy"]:
            answer_t[0, hrf_num - 1] = torch.tensor(vocabulary.convert_tokens_to_ids([answer]))
            hrf_features.append(
                HRF_Features(
                    max_hrf_len=max_seq_length,
                    N=N_t,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    var_in_mask=var_in_mask,
                    answer=answer_t,
                    out_pos=out_pos_t,
                    var_out_mask=var_out_mask,
                    mask_type=mask_type,
                    edge_order=edge_order,
                )
            )

                # # for debug
                # if count > 0:
                #     assert hrf_features[-1].format_str() == last_str
                #     assert last_answer != answer
                #     assert torch.allclose(answer_t, last_answer_t) == False
                # last_str = hrf_features[-1].format_str()
                # last_answer_t = answer_t.clone().detach()
                # last_answer = answer
                # count += 1
    return hrf_features

def VAR2MASK(hrf):
    for id in range(len(hrf)):
        if hrf[id] == "VAR":
            hrf[id] = "[MASK]"
    return hrf


def calc_logic_features(logic_infos, type_id_map, logic_max_arity=4):
    num_hrf = len(logic_infos)
    types_t = torch.zeros(size=[1, num_hrf], dtype=torch.int64)
    in_vars_t = - torch.ones(size=(logic_max_arity, num_hrf), dtype=torch.int64)
    for order, logic_info in enumerate(logic_infos):
        types_t[0, order] = type_id_map[logic_info.type]
        in_vars = logic_info.in_vars
        num_in_vars = len(in_vars)
        in_vars_t[:num_in_vars, order] = torch.tensor(in_vars)
    return Logic_Features(
        max_logic_len=logic_max_arity,
        type=types_t,
        in_vars=in_vars_t,
    )

def get_dataloader_ls(data_reader_dict, run_type_ls, shuffle_config, config):
    dataloader_ls = {}
    for run_type in run_type_ls:
        dataloader_ls[run_type] = DataLoader.DataLoader(
            data_reader_dict[run_type], 
            collate_fn=collate_batch_query,
            batch_size= config["batch_size"], 
            shuffle = shuffle_config[run_type], 
            drop_last=False,
            num_workers=config["num_workers"]
            )
    return dataloader_ls