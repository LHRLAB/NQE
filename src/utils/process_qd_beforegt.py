vocab_size = {
    'wd50k_qe_2000':47688, 
    'wd50k_qe_100':47688, 
    'wd50k_qe': 47688,
    'wd50k_json': 46335,
    "fb15k_237_json": 14981,
    'wd50k_vocab': 47688,
    }
num_relations = {
    'wd50k_qe_2000':531,     
    'wd50k_qe_100':531, 
    'wd50k_qe': 531,
    "wd50k_json": 517,
    "fb15k_237_json": 474,
    'wd50k_vocab': 531,
    }
max_seq_len = {
    'wd50k_qe_2000':7, 
    'wd50k_qe_100':7, 
    'wd50k_qe':7,
    "wd50k_json": 19,
    "fb15k_237_json": 7,
    'wd50k_vocab': 19,
    }
max_arity = {
    'wd50k_qe_2000':4,     
    'wd50k_qe_100':4, 
    'wd50k_qe': 4,
    "wd50k_json": 10,
    "fb15k_237_json": 4,
    'wd50k_vocab': 10,
    }


def process_qd_beforegt(dataset_name, config):
    dataset_name = dataset_name.lower()
    config["dataset_dir"] = "./data/" + dataset_name + "/tasks"
    config["gt_dir"] = "./data/" + dataset_name + "/gt"
    config["vocab_path"] = "./data/" + dataset_name + "/vocab.txt"
    config["vocab_size"] = vocab_size[dataset_name]
    config["num_relations"] = num_relations[dataset_name]
    config["max_seq_len"] = max_seq_len[dataset_name]
    config["max_arity"] = max_arity[dataset_name]

    return config