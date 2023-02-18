import numpy as np
import random
import torch
import time
import os
import json
import logging
import pickle
import wandb
from collections import defaultdict
import time
from new_model.regularizers import *


def get_regularizer(regularizer_setting, entity_dim, neg_input_possible=True, entity=False):
    """
    :param neg_input_possible: for matrix_L1 (class MatrixSumRegularizer)
    :param dual: only apply regularizer to the first half embeddings (after chunk dim=-1) (for sigmoid only)
    """
    if entity:
        key = 'e_reg_type'
    else:
        key = 'type'

    add_layernorm = regularizer_setting['e_layernorm']
    if regularizer_setting[key] == '01':
        regularizer = Regularizer(base_add=0, min_val=0, max_val=1)
    elif regularizer_setting[key] == 'matrix_softmax':
        prob_dim = regularizer_setting['prob_dim']
        regularizer = MatrixSoftmaxRegularizer(entity_dim, prob_dim)
    elif regularizer_setting[key] == 'vector_softmax':
        regularizer = VectorSoftmaxRegularizer(entity_dim)
    elif regularizer_setting[key] == 'sigmoid':
        regularizer = SigmoidRegularizer(entity_dim, dual=regularizer_setting['dual'])
    elif regularizer_setting[key] == 'matrix_L1':
        prob_dim = regularizer_setting['prob_dim']
        regularizer = MatrixSumRegularizer(entity_dim, prob_dim, neg_input_possible)
    elif regularizer_setting[key] == 'matrix_sigmoid_L1':
        prob_dim = regularizer_setting['prob_dim']
        regularizer = MatrixSigmoidSumRegularizer(entity_dim, prob_dim, neg_input_possible)
    elif regularizer_setting[key] == 'vector_sigmoid_L1':
        regularizer = VectorSigmoidSumRegularizer(entity_dim, neg_input_possible, add_layernorm)
    return regularizer

def print_parameters(model):
    print('Model parameters:')
    num_params = 0
    for name, param in model.named_parameters():
        print('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    print('Parameter Number: %d' % num_params)

def read_num_entity_relation_from_file(data_path):
    with open('%s/stats.txt'%data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    return nentity, nrelation


def wandb_initialize(config_dict):
    return wandb.init(
        project="kgfolreasoning",
        # entity='kgfol',
        config=config_dict
    )


def save_model(model, optimizer, save_variable_list, save_dir, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(save_dir, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_dir, 'checkpoint')
    )


def set_logger(args):
    """
    Write logs to console and log file
    """
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        print('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        wandb.log({f'{mode}_{metric}': metrics[metric], 'current_step': step})


def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    average_pos_metrics = defaultdict(float)
    average_neg_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict)
    num_query_structures = 0

    num_pos_query_structures = 0
    num_neg_query_structures = 0

    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode + " " + query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            query_name = query_name_dict[query_structure]  # e.g. 1p
            all_metrics["_".join([query_name, metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
                if 'n' in query_name:
                    average_neg_metrics[metric] += metrics[query_structure][metric]
                else:
                    average_pos_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1
        if 'n' in query_name:
            num_neg_query_structures += 1
        else:
            num_pos_query_structures += 1

    for metric in average_pos_metrics:
        average_pos_metrics[metric] /= num_pos_query_structures
        # writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average_pos", metric])] = average_pos_metrics[metric]

    for metric in average_neg_metrics:
        average_neg_metrics[metric] /= num_neg_query_structures
        # writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average_neg", metric])] = average_neg_metrics[metric]


    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        # writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]

    log_metrics('%s average' % mode, step, average_metrics)
    log_metrics('%s average_pos' % mode, step, average_pos_metrics)
    log_metrics('%s average_neg' % mode, step, average_neg_metrics)


    return all_metrics



def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def flatten_query_and_convert_structure_to_idx(query_structure2queries, query_structure2idx):
    """
    :param query_structure2queries: type dict{query_structure: list[query_info(with entity and relation id)]}
                    e.g. {('e', ('r',)): [(8410, (11,)), (7983, (12,))}
    :param query_structure2idx: type dict{query_structure: structure_idx}
                    e.g. {('e', ('r',)): 0}
    :return all_queries: type list[(query_info, query_structure)]
    """
    all_query_list = [(q, query_structure2idx[query_structure])
                      for query_structure, query_list in query_structure2queries.items()
                      for q in query_list]
    return all_query_list