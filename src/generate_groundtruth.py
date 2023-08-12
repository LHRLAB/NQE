from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cProfile import label

import tarfile
import shutil
import argparse
import multiprocessing
import os

import time
import logging
import numpy as np
import torch
import torch.nn
import torch.optim


from new_reader.generate_gt import generate_gt_dict


from new_reader.vocab_reader import Vocabulary
from utils.args import ArgumentGroup, print_arguments
from utils.process_qd_beforegt import process_qd_beforegt


torch.set_printoptions(precision=8)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())

# yapf: disable
parser = argparse.ArgumentParser()


run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("gpu_index",             int,            2,      "gpu index")
run_type_g.add_arg("do_train",                     bool,   True, "Whether to perform training.")
run_type_g.add_arg("do_predict",                   bool,   True, "Whether to perform prediction.")


# real input dir
parser.add_argument("--dataset",       default="wd50k_qe",   type=str)
parser.add_argument("--train_tasks", default="*", type=str)
parser.add_argument("--validation_tasks", default="*", type=str)
parser.add_argument("--test_tasks", default="*", type=str)
parser.add_argument("--train_shuffle", default=True, type=bool)
parser.add_argument("--prediction_ckpt", default="ckptsWD50K_QE-best-DIM256.ckpt", type=str)

args = parser.parse_args()

def main(args):
    if not (args.do_train or args.do_predict):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")
    config = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)

    if args.use_cuda:
        device = torch.device("cuda")
        config["device"]="cuda"
    else:
        device = torch.device("cpu")
        config["device"]="cpu"

    config = process_qd_beforegt(config['dataset'], config) 

    vocabulary = Vocabulary(
        vocab_file=config["vocab_path"],
        num_relations=config["num_relations"],
        num_entities=config["vocab_size"] - config["num_relations"] - 2)

    # Init program

    train_tasks = config["train_tasks"].split(",")
    validation_tasks = config["validation_tasks"].split(",")
    test_tasks = config["test_tasks"].split(",")
    task_filter = []
    for task in train_tasks:
        if task == "*":
            task_filter = ["*"]
            break
        if task not in task_filter:
            task_filter.append(task)
    for task in validation_tasks:
        if "*" in task_filter:
            task_filter = ["*"]
            break
        if task not in task_filter:
            task_filter.append(task)
    for task in test_tasks:
        if "*" in task_filter:
            task_filter = ["*"]
            break
        if task not in task_filter:
            task_filter.append(task)
    if "*" in task_filter:
        task_filter = ["*"]


    run_type_ls = ["train", "validation", "test"]

    generate_gt_dict(all_task_filter=task_filter, vocabulary=vocabulary, run_type_ls=run_type_ls, config=config)

    return

if __name__ == '__main__':
    print_arguments(args)
    main(args)
