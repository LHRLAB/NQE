import torch 
import torch.nn as nn
from scoring_function.scoring_function import Fuzzy_Score

class Scores_and_Labels(nn.Module):
    def __init__(self, config):
        super(Scores_and_Labels, self).__init__()
        self._emb_size = config['hidden_size']
        self._voc_size = config['vocab_size']
        self._prepostprocess_dropout = config['hidden_dropout_prob']

        self._n_relation = config['num_relations']
        self._n_edge = config['num_edges']
        self._max_seq_len = config['max_seq_len']
        self._max_arity = config['max_arity']
        self._e_soft_label = config['entity_soft_label']
        self._r_soft_label = config['relation_soft_label']

        self._device=config["device"]        
        # self.fc2_bias = torch.nn.init.constant_(torch.nn.parameter.Parameter(torch.Tensor(self._voc_size)), 0.0)
        self.scoring_func = Fuzzy_Score(self._voc_size)

    def forward(self, h_masked, embedding_weight, batch_size, mask_type, mask_label):
        # transform: node embedding weight sharing
        # fc_out=torch.nn.functional.linear(h_masked, self.node_embedding.weight, self.fc2_bias)
        fc_out = self.scoring_func(h_masked, embedding_weight)
        #type_indicator [vocab_size,(yes1 or no0)]
        special_indicator = torch.empty(batch_size,2).to(self._device)
        torch.nn.init.constant_(special_indicator,-1)
        relation_indicator = torch.empty(batch_size, self._n_relation).to(self._device)
        torch.nn.init.constant_(relation_indicator,-1)
        entity_indicator = torch.empty(batch_size, (self._voc_size - self._n_relation - 2)).to(self._device)
        torch.nn.init.constant_(entity_indicator,1)              
        type_indicator = torch.cat((relation_indicator, entity_indicator), dim=1).to(self._device)
        type_indicator = torch.mul(type_indicator, mask_type)
        type_indicator = torch.cat([special_indicator, type_indicator], dim=1)
        type_indicator=torch.nn.functional.relu(type_indicator)
        #排除类型不匹配的
        fc_out_mask=1000000.0*(type_indicator-1.0)
        fc_out = torch.add(fc_out, fc_out_mask)        
        #get one_hot and 候选者（非自身）个数
        one_hot_labels = torch.nn.functional.one_hot(mask_label, self._voc_size)
        type_indicator = torch.sub(type_indicator, one_hot_labels)
        num_candidates = torch.sum(type_indicator, dim=1)
        #get soft label
        soft_labels = ((1 + mask_type) * self._e_soft_label +
                       (1 - mask_type) * self._r_soft_label) / 2.0
        soft_labels=soft_labels.expand(-1,self._voc_size)       
        soft_labels = soft_labels * one_hot_labels + (1.0 - soft_labels) * \
                      torch.mul(type_indicator, 1.0/torch.unsqueeze(num_candidates,1))    
        return fc_out, soft_labels