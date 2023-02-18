vocab_size = {
    'wd50k_qe': 47688,
    'wd50k_nfol': 46335,
    }
num_relations = {
    'wd50k_qe': 531,
    "wd50k_nfol": 517,
    }
max_seq_len = {
    'wd50k_qe':7,
    "wd50k_nfol": 19,
    }
max_arity = {
    'wd50k_qe': 4,
    "wd50k_nfol": 10,
    }



def process_qe_dataset(dataset_name, config):
    dataset_name = dataset_name.lower()
    config["dataset_dir"] = "./data/" + dataset_name + "/tasks"
    config["validation_gt_dir"] = "./data/" + dataset_name + config["relative_validation_gt_dir"]
    config["test_gt_dir"] = "./data/" + dataset_name + config["relative_test_gt_dir"]
    config["vocab_path"] = "./data/" + dataset_name + "/vocab.txt"
    config["vocab_size"] = vocab_size[dataset_name]
    config["num_relations"] = num_relations[dataset_name]
    config["max_seq_len"] = max_seq_len[dataset_name]
    config["max_arity"] = max_arity[dataset_name]

    return config