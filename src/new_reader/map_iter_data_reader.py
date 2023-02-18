from __future__ import print_function
from __future__ import division

import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
# import json
# import ujson as json
import orjson as json
import numpy as np
import logging
import time
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

def new_read_query_graphs_from_file(input_file, max_arity, task):
    query_type = task

    hrf_infos = []
    total_hrf = 0
    arity_stats = np.zeros(shape=max_arity)
    
    logic_str = ["direct", "not", "and", "or"]
    type_id_map = {}
    for id, type_str in enumerate(logic_str):
        type_id_map[type_str] = id
    logic_stats = [0, 0, 0, 0]

    max_seq_len = 2 * ( max_arity - 2 ) + 3
    
    # data = {}
    # total_query = len(data)
    query_ls = []

    time_start = time.time()
    line_count = 0

    with open(input_file, 'r', encoding='utf-8', newline="\n") as fr:
        try:
            while True:
                line_data = fr.readline()
                if line_data:
                    new_dict = json.loads(line_data)
                    for query_idx in new_dict:
                        query_dict = new_dict[query_idx]

                        op_num = len(query_dict)
                        assert op_num % 2 == 0
                        hrf_num = op_num // 2
                        hrf_ls = []
                        logic_ls = []
                        for order in range(hrf_num):
                            hrf_dict = query_dict["hrf" + str(order)]
                            seq_len = hrf_dict["N"] + 1
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

                        query_ls.append(
                            SeqQueryGraph_Info(
                                query_type=query_type,
                                num_hrf=hrf_num,
                                hrf_infos=hrf_ls,
                                logic_infos=logic_ls,
                            )
                        )

                        hrf_infos.extend(hrf_ls)
                        total_hrf += hrf_num

                    line_count += 1
                    if line_count % 5000 == 0:
                        time_now = time.time()
                        time_consumption = time_now - time_start
                        time_sec = time_consumption % 60
                        time_min = ( time_consumption % 3600 ) // 60 
                        time_hour = time_consumption // 3600
                        logger.info("\n %s read lines %d consuming %dh %dm %ss" % (input_file, line_count, time_hour, time_min, time_sec) )
                else:  break
        except Exception as e:
            print(e)
            fr.close()

    total_query = len(query_ls)

    
    return query_ls

def new_query_dataset(vocabulary, run_type_ls, config, task_filters):
    """
    Read a n-ary json file into a list of NaryExample.
    """
    input_dir = config["dataset_dir"]
    max_arity = config["max_arity"]
    max_seq_length = config["max_seq_len"]
    batch_size = config["batch_size"]
    max_dataset_len = config["max_dataset_len"]
    data_reader_dict = {}
    for run_type in run_type_ls:
        tasks_ls = os.listdir(input_dir)
        tasks_ls.sort()
        task_filter = task_filters[run_type]
        if run_type == "train":
            query_datasets_ls = []
            tasks_ls = os.listdir(input_dir)
            tasks_ls.sort()
            for task in tasks_ls:
                if ( task_filter[0] != "*" ) and ( task not in task_filter ):
                    continue
                logger.info("reading queries tasks: %s" % task)
                task_dir = os.path.join(input_dir, task)
                if os.path.isfile(task_dir):
                    continue
                input_file = os.path.join(task_dir, run_type+".json")
                single_dataset = New_Generator_Single_Type_Query_Dataset(max_dataset_len, input_file, task, vocabulary, max_arity, max_seq_length, run_type)
                if ( task_filter[0] == "*" ) or ( task in task_filter ):
                    query_datasets_ls.append(single_dataset)
            data_reader_dict[run_type] = torch.utils.data.ConcatDataset(query_datasets_ls)
        
        else:
            data_reader_dict[run_type] = Sample_Query_Dataset(
                input_dir=input_dir, 
                task_ls=tasks_ls, 
                task_filter=task_filter,
                vocabulary=vocabulary, 
                batch_size=batch_size,
                max_arity=max_arity, 
                max_seq_length=max_seq_length, 
                run_type=run_type,
                )
    return data_reader_dict


class Sample_Query_Dataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        input_dir, 
        task_ls, 
        task_filter, 
        vocabulary:Vocabulary,
        batch_size:int, 
        max_arity=4, 
        max_seq_length=7, 
        run_type="train",
        sampler_weights=None):
        super(Sample_Query_Dataset, self).__init__()
        self.input_dir = input_dir 
        self.task_ls = task_ls
        self.task_filter = task_filter
        self.run_type = run_type
        self.vocabulary=vocabulary
        self.max_arity=max_arity
        self.max_seq_length=max_seq_length
        self.batch_size = batch_size

        self.task_iter_datasets = []
        for task in self.task_ls:

            if ( self.task_filter[0] != "*" ) and ( task not in self.task_filter ):
                continue

            task_dir = os.path.join(self.input_dir, task)
            if os.path.isfile(task_dir):
                continue
            # tasks.append(task)
            input_file = os.path.join(task_dir, self.run_type+".json")

            self.task_iter_datasets.append(
                Single_Iter_Query_Dataset(
                    input_file=input_file,
                    task=task,
                    vocabulary=self.vocabulary,
                    max_arity=self.max_arity,
                    max_seq_length=self.max_seq_length,
                    run_type=self.run_type
                )
            )
        self.task_num = len(self.task_iter_datasets)
        
        if sampler_weights == None:
            self.sampler_weights = np.ones(shape=self.task_num, dtype=np.int64)
        else:
            self.sampler_weights = sampler_weights 

        self._type_sampler = Weighted_Sampler(self.sampler_weights)

    def __iter__(self):
        task_id = self._type_sampler.sample()        
        return self.task_iter_datasets[task_id].read_next()

class Weighted_Sampler:
    def __init__(self, sampler_weights):
        super(Weighted_Sampler).__init__()
        self.type_num = len(sampler_weights)
        self.weights = np.array(sampler_weights, dtype=np.int64)
        self.res_type_map = - np.ones(shape=self.type_num, dtype=np.int64)
        start = 0
        for type_id, weight in enumerate(self.weights):
            self.res_type_map[start: start+weight] = type_id 
            start += weight
        self.choice_num = start

    def sample(self):
        choice = np.random.choice(self.choice_num)
        return self.res_type_map[choice]

    def sample_batch_size(self, batch_size):
        choices = np.random.choice(self.choice_num, batch_size)
        return self.res_type_map[choices]

class Single_Iter_Query_Dataset:
    def __init__(self, input_file, task, vocabulary, max_arity, max_seq_length, run_type):
        super(Single_Iter_Query_Dataset, self).__init__()
        self.input_file = input_file 
        self.query_type = task 
        self.vocabulary = vocabulary 
        
        assert max_seq_length == 2 * ( max_arity - 2 ) + 3

        self.max_arity = max_arity 
        self.max_seq_length = max_seq_length 
        self.run_type = run_type

        # total_hrf = 0
        self.arity_stats = np.zeros(shape=(self.max_arity+1))
        
        self.logic_str = ["direct", "not", "and", "or"]
        self.type_id_map = {}
        for id, type_str in enumerate(self.logic_str):
            self.type_id_map[type_str] = id
        self.logic_stats = [0, 0, 0, 0]
        self.count = 0

    def read_next(self):     
        logger.info("query type = %s" % self.query_type)

        time_start = time.time()

        with open(self.input_file, 'r', encoding='utf-8', newline="\n") as fr:
            while True:
                line_data = fr.readline()
                if line_data:
                    new_dict = json.loads(line_data)
                    for query_idx in new_dict:
                        query_dict = new_dict[query_idx]
                        len_bigger = False
                        idx = int(query_idx.split("_")[-1])
                        if idx % 100 == 0: logger.info("query %s - %d" % (self.query_type, idx) )
                        op_num = len(query_dict)
                        assert op_num % 2 == 0
                        hrf_num = op_num // 2
                        hrf_ls = []
                        logic_ls = []
                        for order in range(hrf_num):
                            hrf_id = "hrf" + str(order)
                            hrf_dict = query_dict[hrf_id]
                            seq_len = hrf_dict["N"] + 1

                            if seq_len > self.max_seq_length:
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

                            self.arity_stats[arity] += 1
                            
                            logic_id = "logic" + str(order)     
                            logic_dict = query_dict[logic_id]       
                            logic_info = Logic_Info(
                                type=logic_dict["type"],
                                in_vars=logic_dict["in_vars"],
                            )

                            self.logic_stats[self.type_id_map[logic_dict["type"]]] += 1

                            hrf_ls.append(hrf_info)
                            logic_ls.append(logic_info)

                        if len_bigger:
                            break

                        hrf_features_ls = calc_hrf_features(hrf_ls, self.vocabulary, self.max_arity, self.max_seq_length)
                        logic_str = ["direct", "not", "and", "or"]
                        logic_features = calc_logic_features(logic_ls, self.type_id_map)
                        for hrf_features in hrf_features_ls:

                            self.count += 1
                            if self.count % 500 == 0:
                                time_now = time.time()
                                time_consumption = time_now - time_start
                                time_sec = time_consumption % 60
                                time_min = ( time_consumption % 3600 ) // 60 
                                time_hour = time_consumption // 3600
                                logger.info( \
                                    "query_type: %s-%s read queries: %d time_consumption: %dh %dm %.2fs"  % \
                                        (self.query_type, self.run_type, self.count, time_hour, time_min, time_sec) )

                            seq_feat = SeqQueryGraph_Features(
                                query_type=self.query_type,
                                num_hrf=hrf_num,
                                hrf_features=hrf_features,
                                logic_features=logic_features,
                            )
                            yield seq_feat
                else: return

class New_Generator_Single_Type_Query_Dataset(Dataset.Dataset):
    def __init__(self, max_dataset_len, input_file, task, vocabulary:Vocabulary, max_arity=4, max_seq_length=7, run_type="train"):
        self.input_file = input_file 
        self.task = task
        self.vocabulary=vocabulary
        self.max_arity=max_arity
        self.max_seq_length=max_seq_length
        self.run_type = run_type

        self.query_features = list(
            itertools.islice(
                generator_convert_info_to_features(
                    input_file=self.input_file,
                    task=self.task,
                    vocabulary=self.vocabulary,
                    max_arity=self.max_arity,
                    max_seq_length=self.max_seq_length,
                    run_type=self.run_type,
                ),
                max_dataset_len,
            )
        )

    def __len__(self):
        return len(self.query_features)

    def __getitem__(self, index):        
        x = self.query_features[index]
        return x


def convert_info_to_features(query_ls, vocabulary, max_arity, max_seq_length):

    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, \
        "Each input sequence contains relation, head, tail, " \
        "and max_aux attribute-value pairs."

    query_features_ls = []
    for (query_id, query) in enumerate(query_ls):
        query_type = query.query_type
        num_hrf = query.num_hrf
        hrf_infos = query.hrf_infos
        logic_infos = query.logic_infos
        hrf_features_ls = calc_hrf_features(hrf_infos, vocabulary, max_arity, max_seq_length)
        logic_str = ["direct", "not", "and", "or"]
        type_id_map = {}
        for id, type_str in enumerate(logic_str):
            type_id_map[type_str] = id
        logic_features = calc_logic_features(logic_infos, type_id_map)
        for hrf_features in hrf_features_ls:
            query_features_ls.append(
                SeqQueryGraph_Features(
                    query_type=query_type,
                    num_hrf=num_hrf,
                    hrf_features=hrf_features,
                    logic_features=logic_features,
                )
            ) 
    return query_features_ls
        
def generator_convert_info_to_features(input_file, task, vocabulary, max_arity, max_seq_length, run_type):
    query_type = task
                        
    logger.info("query type = %s" % query_type)

    total_hrf = 0
    arity_stats = np.zeros(shape=max_arity)
    
    logic_str = ["direct", "not", "and", "or"]
    type_id_map = {}
    for id, type_str in enumerate(logic_str):
        type_id_map[type_str] = id
    logic_stats = [0, 0, 0, 0]

    max_seq_length = 2 * ( max_arity - 2 ) + 3

    time_start = time.time()
    count = 0
    with open(input_file, 'r', encoding='utf-8', newline="\n") as fr:
        try:
            while True:
                wrong_item = False
                line_data = fr.readline()
                if line_data:
                    new_dict = json.loads(line_data)

                    for query_idx in new_dict:

                        query_dict = new_dict[query_idx]

                        # logger.info("query index = %s" % query_idx)

                        op_num = len(query_dict)
                        if op_num % 2 != 0:
                            wrong_item=True
                            break                            
                        hrf_num = op_num // 2
                        hrf_ls = []
                        logic_ls = []
                        for order in range(hrf_num):
                            hrf_id = "hrf" + str(order)
                            hrf_dict = query_dict[hrf_id]
                            seq_len = hrf_dict["N"] + 1
                            if seq_len % 2 != 1:
                                wrong_item=True
                                break                                
                            arity = ( seq_len + 1 ) // 2
                            if arity >= max_arity:
                                wrong_item=True
                                break
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
                        if wrong_item:
                            break

                        hrf_features_ls = calc_hrf_features(hrf_ls, vocabulary, max_arity, max_seq_length)
                        logic_str = ["direct", "not", "and", "or"]
                        type_id_map = {}
                        for id, type_str in enumerate(logic_str): type_id_map[type_str] = id
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

                            seq_feat = SeqQueryGraph_Features(
                                query_type=query_type,
                                num_hrf=hrf_num,
                                hrf_features=hrf_features,
                                logic_features=logic_features,
                            )
                            
                            yield seq_feat
                    if wrong_item:
                        continue
                else: 
                    break
        except Exception as e:
            print(e)
            fr.close()

class Generator_Single_Type_Query_Dataset(Dataset.Dataset):
    def __init__(self, vocabulary:Vocabulary, query_ls, max_arity=4, max_seq_length=7):
        self.query_ls = query_ls
        self.vocabulary=vocabulary
        self.max_arity=max_arity
        self.max_seq_length=max_seq_length

        self.query_features = list(
            itertools.islice(
                generator_convert_info_to_features(
                    query_ls=self.query_ls,
                    vocabulary=self.vocabulary,
                    max_arity=self.max_arity,
                    max_seq_length=self.max_seq_length,
                ),
                None,
            )
        )

    def __len__(self):
        return len(self.query_features)

    def __getitem__(self, index):        
        x = self.query_features[index]
        return x





def generator_reader_features(query_ls, vocabulary, task_filter, max_arity, max_seq_length):
    all_query_dataset_ls = []
    for query_type in query_ls:
        if ( task_filter[0] != "*" ) and ( query_type not in task_filter ):
            continue
        single_type_query_ls = query_ls[query_type]
        all_query_dataset_ls.append(Generator_Single_Type_Query_Dataset(vocabulary, single_type_query_ls, max_arity, max_seq_length))
    return torch.utils.data.ConcatDataset(all_query_dataset_ls), all_query_dataset_ls

def collate_batch_query(queries_features: Sequence[SeqQueryGraph_Features]):
    query_types, query_strs, input_ids_ls, input_mask_ls, edge_orders_ls, hrf_start_ls, var_in_mask_ls = [],[],[],[],[],[],[]
    var_out_mask_ls, out_pos_ls, answers_ls, mask_type_ls, logic_in_ids_ls, logic_types_ls, batch_answer_mask_ls = [],[],[],[],[],[],[]

    edge_offset = 0
    for local_id, query_features in enumerate(queries_features):
        query_type = query_features.query_type
        query_str = query_features.format_str()
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

        query_types.append(query_type)
        query_strs.append(query_str)
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
        query_types=query_types,
        query_strs=query_strs,
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

def get_dataloader_ls(data_reader_dict, run_type_ls, config):
    dataloader_ls = {}
    for run_type in run_type_ls:
        if run_type == "train": do_shuffle = config["train_shuffle"]
        else: do_shuffle = False
        dataloader_ls[run_type] = DataLoader.DataLoader(
            data_reader_dict[run_type], 
            collate_fn=collate_batch_query,
            batch_size= config["batch_size"], 
            shuffle = do_shuffle,             
            drop_last=False,
            num_workers=config["num_workers"]
            )
    return dataloader_ls