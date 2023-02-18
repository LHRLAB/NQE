
import torch
import dataclasses
from typing import Dict, List

LongTensor = torch.Tensor


@dataclasses.dataclass()
class HRF_Info:
    N: int 
    hrf: List[str]
    out_pos: int 
    answers: Dict
    in_pos: List[int]
    in_vars: List[int]

    def __post_init__(self):
        assert self.N >= 2
        assert self.N <= 18
        # assert self.out_pos == 2 
        assert len(self.answers) >= 1
        assert len(self.in_pos) == len(self.in_vars)

    def format_out(self):
        hrf_dict = {}
        hrf_dict["N"] = self.N 
        hrf_dict["hrf"] = self.hrf 
        hrf_dict["out_pos"] = self.out_pos 
        hrf_dict["answers"] = self.answers 
        hrf_dict["in_pos"] = self.in_pos 
        hrf_dict["in_vars"] = self.in_vars 
        return hrf_dict

@dataclasses.dataclass
class HRF_Features:
    def __init__(
        self,
        max_hrf_len: int,
        N: LongTensor,
        input_ids: LongTensor,
        input_mask: LongTensor,
        var_in_mask: LongTensor,
        answer: LongTensor,
        out_pos: LongTensor,
        var_out_mask: LongTensor,
        mask_type: LongTensor,
        edge_order: LongTensor,
    ):
        self.max_hrf_len = max_hrf_len
        self.N = N.clone().detach()
        self.input_ids = input_ids.clone().detach()
        self.input_mask = input_mask.clone().detach()
        self.var_in_mask = var_in_mask.clone().detach()
        self.answer = answer.clone().detach()
        self.out_pos = out_pos.clone().detach()
        self.var_out_mask = var_out_mask.clone().detach()
        self.mask_type = mask_type.clone().detach()
        self.edge_order = edge_order.clone().detach()         

        assert self.input_ids.shape[0] == self.max_hrf_len
        assert self.input_ids.shape == self.input_mask.shape

    def format_str(self):
        return flatten_list_str(
            [
                self.N, 
                self.input_ids,
                self.input_mask,
                self.var_in_mask,
                self.out_pos,
                self.var_out_mask,
                self.edge_order,
            ]
        )


@dataclasses.dataclass()
class Logic_Info:
    type: str 
    in_vars: List[int]

    def __post_init__(self):
        assert self.type in ["direct", "not", "and", "or"]
        assert len(self.in_vars) >= 1

    def format_out(self):
        logic_dict = {}
        logic_dict["type"] = self.type 
        logic_dict["in_vars"] = self.in_vars
        return logic_dict


@dataclasses.dataclass
class Logic_Features:
    def __init__(
        self,    
        max_logic_len: int,
        type: LongTensor,
        in_vars: LongTensor,
    ):
        self.max_logic_len = max_logic_len 
        self.type = type.clone().detach()
        self.in_vars = in_vars.clone().detach()

        assert self.in_vars.shape[0] == self.max_logic_len

    def format_str(self):
        return flatten_list_str([self.type, self.in_vars])

@dataclasses.dataclass
class SeqQueryGraph_Info:
    query_type: str
    num_hrf: int 
    hrf_infos: List[HRF_Info]
    logic_infos: List[Logic_Info]


@dataclasses.dataclass
class SeqQueryGraph_Features:
    """A sequence representing query graphs."""
    query_type: str 
    num_hrf: int
    hrf_features: HRF_Features
    logic_features: Logic_Features
    
    def format_str(self):
        hrf_str = self.hrf_features.format_str() 
        logic_str = self.logic_features.format_str()
        return hrf_str + logic_str

    def get_query_answer(self):
        answer_mask = self.hrf_features.answer
        return answer_mask[answer_mask>=0].item()

@dataclasses.dataclass
class SeqQueryGraphBatch:
    """A batch of sequence query graphs."""
    def __init__(
        self,
        input_ids: LongTensor,
        input_mask: LongTensor,
        edge_orders: LongTensor,
        hrf_start: LongTensor,
        var_in_mask: LongTensor,
        var_in_batch_mask: LongTensor,
        var_out_mask: LongTensor,
        var_out_batch_mask: LongTensor,
        out_pos: LongTensor,
        answers: LongTensor,
        mask_type: LongTensor,
        logic_in_ids: LongTensor,
        logic_in_batch_ids: LongTensor,
        logic_types: LongTensor,
        batch_answer_mask: LongTensor,
        query_types: List[str],
        query_strs: List[str],
    ):
        self.input_ids = input_ids.clone().detach()
        self.input_mask = input_mask.clone().detach()
        self.edge_orders = edge_orders.clone().detach()
        self.hrf_start = hrf_start.clone().detach()
        self.var_in_mask = var_in_mask.clone().detach()
        self.var_in_batch_mask = var_in_batch_mask.clone().detach()
        self.var_out_mask = var_out_mask.clone().detach()
        self.var_out_batch_mask = var_out_batch_mask.clone().detach()
        self.out_pos = out_pos.clone().detach()
        self.answers = answers.clone().detach()
        self.mask_type = mask_type.clone().detach()
        self.logic_in_ids = logic_in_ids.clone().detach()
        self.logic_in_batch_ids = logic_in_batch_ids.clone().detach()
        self.logic_types = logic_types.clone().detach()
        self.batch_answer_mask = batch_answer_mask.clone().detach()  
        self.query_types = query_types
        self.query_strs = query_strs

        assert self.input_ids is not None

def flatten_list_str(data_ls):
    out_str = ""
    for data in data_ls:
        out_str += str(data.flatten().tolist())
    return out_str