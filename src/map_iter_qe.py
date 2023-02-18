from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cProfile import label

import tarfile
import shutil
import argparse
import multiprocessing
import os

import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn
import torch.optim


from new_reader.vocab_reader import Vocabulary
from new_reader.map_iter_data_reader import new_query_dataset, get_dataloader_ls
from new_model.query_embedding import NGEModel
from new_evaluation.iter_qe_evaluation import load_ground_truth, batch_evaluation, compute_metrics
from utils.args import ArgumentGroup, print_arguments
from utils.process_qe_dataset import process_qe_dataset

import collections

torch.set_printoptions(precision=8)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())

parser = argparse.ArgumentParser()

model_g = ArgumentGroup(parser, "model", "model and checkpoint configuration.")
model_g.add_arg("num_hidden_layers",       int,    16,        "Number of hidden layers.")
model_g.add_arg("num_attention_heads",     int,    8,         "Number of attention heads.")
model_g.add_arg("hidden_size",             int,    256,       "Hidden size.")#256
model_g.add_arg("intermediate_size",       int,    512,       "Intermediate size.")#512
model_g.add_arg("hidden_dropout_prob",     float,  0.0,       "Hidden dropout ratio.")
model_g.add_arg("attention_dropout_prob",  float,  0.0,       "Attention dropout ratio.")
model_g.add_arg("num_edges",               int,    14,
                "Number of edge types, typically fixed to 5: no edge (0), relation-subject (1),"
                "relation-object (2), relation-attribute (3), attribute-value (4).")
model_g.add_arg("entity_soft_label",       float,  0.8,       "Label smoothing rate for masked entities.")
model_g.add_arg("relation_soft_label",     float,  1.0,       "Label smoothing rate for masked relations.")
model_g.add_arg("checkpoint_dir",             str,    "./src/new_ckpts",   "Path to save checkpoints.")
model_g.add_arg("eval_dir",             str,    "./src/eval_result",   "Path to save eval_result.")
model_g.add_arg("encoder_order",            str,    "L",       "encoder_order") # true_GL L GL
model_g.add_arg("L_config",                 str,        "L_node2_edge",       "L encoder config")# L_none L_node L_edge L_node_edge

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("batch_size",        int,    256,                   "Batch size.")#1024
train_g.add_arg("epoch",             int,    300,                    "Number of training epochs.")
train_g.add_arg("learning_rate",     float,  5e-4,                   "Learning rate with warmup.")#5e-4
train_g.add_arg("weight_decay",      float,  0.01,                   "Weight decay rate for L2 regularizer.")#0.01
train_g.add_arg("save_steps",          int,   5,                  "save_steps")#20
train_g.add_arg("num_workers",      int,    0,                     "num_workers")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    100,    "Step intervals to print loss.")
log_g.add_arg("start_evaluation_epochs",      int,    1, "epoch index to start evaluation")
log_g.add_arg("interval_evaluation_epochs",         int,   10,      "epoch interval for evaluation")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("gpu_index",             int,            0,      "gpu index")
run_type_g.add_arg("do_learn",                     bool,   True, "Whether to perform training.")
run_type_g.add_arg("do_predict",                   bool,   False, "Whether to perform prediction.")

# fuzzy logic
parser.add_argument('--logic_type', default='godel', type=str, choices=['luka', 'godel', 'product', 'godel_gumbel'],
                    help='fuzzy logic type')
parser.add_argument('--regularizer', default='sigmoid', type=str,
                    choices=['01', 'vector_softmax', 'matrix_softmax', 'matrix_L1', 'matrix_sigmoid_L1','sigmoid', 'vector_sigmoid_L1'],
                    help='ways to regularize parameters')
parser.add_argument('--e_regularizer', default='same', type=str,
                    choices=['same', '01', 'vector_softmax', 'matrix_softmax', 'matrix_L1', 'matrix_sigmoid_L1','sigmoid', 'vector_sigmoid_L1'],
                    help='set regularizer for entities, different from queries') 
parser.add_argument('--entity_ln_before_reg', action="store_true", help='apply layer normalization before applying regularizer to entities')
parser.add_argument('-k', '--prob_dim', default=8, type=int, help="for matrix_softmax and matrix_L1. dims per prob vector")
parser.add_argument('--loss_type', default='cos', type=str, help="loss type")

# real input dir

parser.add_argument("--dataset",       default="wd50k_qe",   type=str)
parser.add_argument("--max_dataset_len", default=1000000, type=int)
parser.add_argument("--train_tasks", default="1p", type=str)
parser.add_argument("--validation_tasks", default="1p", type=str)
parser.add_argument("--valid_eval_tasks", default="1p", type=str)
parser.add_argument("--test_tasks", default="1p", type=str)
parser.add_argument("--prediction_tasks", default="1p, 2p, 3p, 2i, 3i, pi, ip", type=str)

# parser.add_argument("--dataset",       default="wd50k_nfol",   type=str)
# parser.add_argument("--max_dataset_len", default=1000000, type=int)
# parser.add_argument("--train_tasks", default="1p", type=str)
# parser.add_argument("--validation_tasks", default="1p", type=str)
# parser.add_argument("--valid_eval_tasks", default="1p", type=str)
# parser.add_argument("--test_tasks", default="1p", type=str)
# parser.add_argument("--prediction_tasks", default="1p, 2p, 3p, 2i, 3i, pi, ip, 2u, up, 2cp, 3cp, 2in, 3in, inp, pin, pni", type=str)

parser.add_argument("--train_shuffle", default=True, type=bool)
parser.add_argument("--prediction_ckpt", default="ckptswd50k_qe-train_tasks-1p-best-valid-DIM256.ckpt", type=str)
parser.add_argument("--relative_validation_gt_dir", default="/gt", type=str)
parser.add_argument("--relative_test_gt_dir", default="/gt", type=str)
args = parser.parse_args()

def main(args):
    if not os.path.exists("log"):
        os.mkdir("log")    
        if not os.path.exists("log/validation"):
            os.mkdir("log/validation")   
        if not os.path.exists("log/test"):
            os.mkdir("log/test")   
            
    if not (args.do_learn or args.do_predict):
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

    config = process_qe_dataset(config['dataset'], config) 

    vocabulary = Vocabulary(
        vocab_file=config["vocab_path"],
        num_relations=config["num_relations"],
        num_entities=config["vocab_size"] - config["num_relations"] - 2)

    warmup_steps = 10000
    max_train_steps = 1000000000

    writer_dir = {
        "validation": "log/validation",
        "test": "log/test",
    }
    log_prefix = {
        "validation": "----- prediction on validation set: ------\n",
        "test": "----- prediction on test set: ------\n",
    }

    if config["do_learn"]:
        learning_run_type_ls = ["train", "validation", "test"]
        task_filters = {}
        for run_type in learning_run_type_ls:
            task_filter = [task.strip() for task in config[run_type+"_tasks"].split(",")]
            task_filters[run_type] = task_filter

        learning_all_run_gt_dict = collections.defaultdict( lambda: collections.defaultdict(lambda: collections.defaultdict(set)))
        learning_all_run_gt_dict["validation"] = load_ground_truth(config["validation_gt_dir"], task_filters["validation"])
        learning_all_run_gt_dict["test"] = load_ground_truth(config["test_gt_dir"], task_filters["test"])

        max_valid_score = 0.000
        valid_score_task_filter = [task.strip() for task in config["valid_eval_tasks"].split(",")]

        learning_data_reader_dict = new_query_dataset(
            vocabulary=vocabulary, 
            run_type_ls=learning_run_type_ls, 
            config=config,
            task_filters=task_filters,
            )
        learning_dataloader_ls = get_dataloader_ls(learning_data_reader_dict, learning_run_type_ls, config)

        gran_model = NGEModel(config=config).to(device)
        optimizer=torch.optim.Adam(gran_model.parameters(),lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps:steps/warmup_steps if steps<=warmup_steps else (max_train_steps-steps)/(max_train_steps-warmup_steps))

        steps = 0
        best_step=0
        best_loss=10000.0
        total_cost=0.0
        average_loss=0.0
        train_total_queries_num = 0

        training_range = tqdm(range(args.epoch))
        for epoch in training_range:          
            for batch_query in learning_dataloader_ls["train"]:
                
                current_batch_size = len(batch_query.query_types)
                logger.info("current batch has %d queries" % current_batch_size)
                steps+=1
                scheduled_lr=optimizer.state_dict()['param_groups'][0]['lr']
                print("scheduled_lr", scheduled_lr)
                gran_model.train() 
                scheduled_lr=optimizer.state_dict()['param_groups'][0]['lr']
                optimizer.zero_grad()
                loss, fc_out = gran_model(batch_query)               
                loss.backward()
                optimizer.step()
                scheduler.step()
                if args.weight_decay >= 0:
                    for param in gran_model.parameters():
                        if param.requires_grad:
                            param_copy = param.data.detach()
                            param.data = param_copy - param_copy * args.weight_decay * scheduled_lr

                gran_model.eval()
                with torch.no_grad():
                    total_cost+=loss
                    average_loss=total_cost/steps                    
                    if loss < best_loss and steps>10:
                        best_loss=loss
                        best_step=steps
                        torch.save(gran_model.state_dict(),os.path.join(args.checkpoint_dir , "ckpts"+args.dataset +"-best-loss-DIM"+str(args.hidden_size)+ ".ckpt"))                   
                    training_range.set_description("Epoch %d | Steps %d | lr: %f | loss: %f | best_loss: %f at step%d | average_loss: %f "  % (epoch,steps, scheduled_lr,loss,best_loss,best_step,average_loss))

                train_total_queries_num += current_batch_size
                logger.info("Model has got %d queries Trained" % train_total_queries_num)

            if epoch % config["interval_evaluation_epochs"] == 0 and epoch >= config["start_evaluation_epochs"]:

                valid_score = print_evaluation(
                    gran_model, 
                    learning_dataloader_ls["validation"], 
                    learning_all_run_gt_dict["validation"], 
                    device, 
                    writer_dir["validation"], 
                    writer_dir["validation"],
                    valid_score_task_filter,
                    )
                if valid_score > max_valid_score:
                    torch.save(gran_model.state_dict(),os.path.join(args.checkpoint_dir , "ckpts"+args.dataset+"-train_tasks-"+config["train_tasks"]+"-best-valid-DIM"+str(args.hidden_size)+ ".ckpt"))
                    max_valid_score = valid_score

        torch.save(gran_model.state_dict(),os.path.join(args.checkpoint_dir , "ckpts"+args.dataset +"-last-DIM"+str(args.hidden_size)+ ".ckpt"))            

    if config["do_predict"]:
        prediction_run_type_ls = ["test", "validation", ]

        gran_model = NGEModel(config=config).to(device)
        gran_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir , config["prediction_ckpt"]))) 
        num_params = sum(param.numel() for param in gran_model.parameters())
        logger.info("Model Parameters Number %d " % num_params)
        task_filters = {}
        for run_type in prediction_run_type_ls:
            task_filter = [task.strip() for task in config["prediction_tasks"].split(",")]
            for task in task_filter:
                task_filters = {}
                task_filters[run_type] = [task]
                single_run_single_task_prediction_data_reader_dict = new_query_dataset(
                    vocabulary=vocabulary, 
                    run_type_ls=[run_type], 
                    config=config,
                    task_filters=task_filters,
                    )
                prediction_dataloader_ls = get_dataloader_ls(
                    data_reader_dict=single_run_single_task_prediction_data_reader_dict, 
                    run_type_ls=[run_type], 
                    config=config
                    )

                prediction_all_run_gt_dict = collections.defaultdict( lambda: collections.defaultdict(lambda: collections.defaultdict(set)))
                prediction_all_run_gt_dict[run_type] = load_ground_truth(config[run_type+"_gt_dir"], task_filters[run_type])
                print_evaluation(
                    gran_model, 
                    prediction_dataloader_ls[run_type], 
                    prediction_all_run_gt_dict[run_type], 
                    device, 
                    writer_dir[run_type], 
                    writer_dir[run_type],
                    [task],
                    )

def evaluate(model, eval_dataloader, single_run_gt_dict, device):

    logger.info("Start evaluating !")
    if not os.path.exists(args.eval_dir): os.makedirs(args.eval_dir)
    eval_result_file = os.path.join(args.eval_dir, "eval_result.json")
    
    task_ranks = collections.defaultdict(list)
    task_left_ids = collections.defaultdict(list)
    task_filter = 0.25
    filtered_task_ranks = collections.defaultdict(list)

    step = 0
    global_idx = 0
    model.eval()
    with torch.no_grad():
        predict_range=tqdm(enumerate(eval_dataloader))
        for i, batch_query in predict_range:
            batch_results = []
            _,np_fc_out = model(batch_query)
            batch_results = np_fc_out.cpu().numpy()

            task_ranks = batch_evaluation(task_ranks, batch_query, batch_results, single_run_gt_dict)
            predict_range.set_description("Processing prediction steps: %d | examples: %d" % (step, global_idx))
            step += 1
            global_idx += np_fc_out.size(0)

    for task in task_ranks:
        cur_task_ranks = np.array(task_ranks[task])
        rank_num = cur_task_ranks.shape[0]
        rank_sort_order = np.argsort(cur_task_ranks)
        task_left_ids[task] = rank_sort_order[ : int( task_filter * rank_num)]
        filtered_task_ranks[task] = cur_task_ranks[task_left_ids[task]]

    eval_metrics = compute_metrics(
        task_ranks=task_ranks,
        eval_metrics_file=eval_result_file,
    )
    logger.info("Evaluate total %d queries Done !" % global_idx)
    return eval_metrics

def print_evaluation(gran_model, eval_dataloader, single_run_gt_dict, device, writer_dir, log_prefix, eval_result_tasks_filter):

    vocab_emb = gran_model.node_embedding.weight.data.cpu().numpy()
    np.save(os.path.join(writer_dir, "vocab_emb"), vocab_emb) 

    eval_metrics = evaluate(
        model=gran_model,
        eval_dataloader=eval_dataloader,
        single_run_gt_dict=single_run_gt_dict,
        device=device
    )

    format_str = ""
    for task in eval_metrics:
        format_str += task  
        for metric_type in eval_metrics[task]: format_str += "\t%.4f" % (eval_metrics[task][metric_type])
        format_str += "\n"         

    logger.info("\n" + log_prefix + "\n")
    
    logger.info("\n-------- Evaluation Performance --------\n%s\n%s" % (
        "\t".join(["TASK", "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]), format_str))

    mrr_scores = []
    for task in eval_result_tasks_filter:
        mrr_scores.append(eval_metrics[task]["mrr"])
        logger.info("calculate MRR scores containing task %s" % task)
    mrr_arr = np.array(mrr_scores)
    avg_mrr_score = np.mean(mrr_arr)
    return avg_mrr_score

if __name__ == '__main__':
    print_arguments(args)
    main(args)
