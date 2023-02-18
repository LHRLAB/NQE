from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from sys import stdin
import torch
import torch.nn
from new_model.graph_encoder import truncated_normal
import numpy as np

from loss.loss import softmax_with_cross_entropy
from scoring_function.calc_scores_and_labels import Scores_and_Labels
from new_model.HAHE import Sequence_Encoder
from new_model.logic_operations import Negation, Conjunction, Disjunction
from new_reader.new_data_structure import SeqQueryGraphBatch

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())




class NGEModel(torch.nn.Module):
    def __init__(self, config):
        super(NGEModel,self).__init__()

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

        self._encoder_order = config['encoder_order']

        #self.node_embedding [V,N*H]
        self.node_embedding=torch.nn.Embedding(self._voc_size, self._emb_size)
        self.node_embedding.weight.data=truncated_normal(self.node_embedding.weight.data,std=0.02)
        self.act = torch.nn.Sigmoid()        
        
        self.seq_enc = Sequence_Encoder(config)

        self.logic_type = config['logic_type']
        self.regularizer_setting={
            'type': config["regularizer"],  # for query
            'e_reg_type': config["regularizer"] if config["e_regularizer"] == 'same' else config["e_regularizer"],
            'prob_dim': config["prob_dim"],  # for matrix softmax
            'dual': True if config["loss_type"] == 'weighted_fuzzy_containment' else False,
            'e_layernorm': config["entity_ln_before_reg"]  # apply Layer Norm before next step's regularizer
        }
        self.conjunction_net = Conjunction(self._emb_size, self.logic_type, self.regularizer_setting,)
        self.disjunction_net = Disjunction(self._emb_size, self.logic_type, self.regularizer_setting,)
        self.negation_net = Negation(self._emb_size, self.logic_type, self.regularizer_setting)

        self.scores_labels = Scores_and_Labels(config)
        self.myloss = softmax_with_cross_entropy()
        #self.myloss = torch.nn.CrossEntropyLoss()
        
    def forward(
        self, 
        seq_batch: SeqQueryGraphBatch,
        ):  

        input_ids = seq_batch.input_ids.to(self._device) 
        input_mask = seq_batch.input_mask.float().to(self._device)

        edge_orders = seq_batch.edge_orders.squeeze().to(self._device) 
        hrf_start = seq_batch.hrf_start.to(self._device) 

        var_in_mask = seq_batch.var_in_mask.to(self._device) 
        var_in_batch_mask = seq_batch.var_in_batch_mask.to(self._device) 
        var_out_mask = seq_batch.var_out_mask.to(self._device) 
        var_out_batch_mask = seq_batch.var_out_batch_mask.to(self._device) 

        out_pos = seq_batch.out_pos.to(self._device) 
        answers = seq_batch.answers.squeeze().to(self._device)  
        mask_type = seq_batch.mask_type.squeeze().to(self._device) 

        logic_in_ids = seq_batch.logic_in_ids.to(self._device) 
        logic_in_batch_ids = seq_batch.logic_in_batch_ids.to(self._device) 
        logic_types = seq_batch.logic_types.squeeze().to(self._device) 

        batch_answer_mask = seq_batch.batch_answer_mask.squeeze().to(self._device)

        max_local_edge_order = torch.max(edge_orders)
        seq_emb = self.node_embedding.weight[input_ids]
        num_edges = input_ids.shape[-1]
        var_emb = torch.ones(size=[2, num_edges, self._emb_size], device=self._device)

        for order in range(max_local_edge_order.item()+1):
            current_order_mask = np.isin(edge_orders.cpu(), order)

            seq_emb[var_in_mask>=0] = var_emb[1, var_in_batch_mask[var_in_mask>=0], :]

            current_seq_emb = seq_emb[:, current_order_mask, :]
            # current_seq_emb = current_seq_emb.transpose(0, 1)
            current_input_mask = input_mask[:, current_order_mask] 
            # current_var_in_mask = var_in_mask[:, current_order_mask]
            # current_seq_emb[current_var_in_mask>0] = var_emb[:, current_var_in_mask>0, :]
            current_var_out_mask = var_out_mask[:, current_order_mask]
            current_out_pos = out_pos[:, current_order_mask]

            enc_out = self.seq_enc(
                current_seq_emb.transpose(0, 1), 
                current_input_mask.transpose(0, 1).unsqueeze(-1), 
                current_out_pos.transpose(0, 1),
                )

            # enc_out = self.act(enc_out)
            var_emb[0, current_order_mask, :] = enc_out
            var_emb[1, current_order_mask, :] = var_emb[0, current_order_mask, :] 

            negation_out = self.negation(var_emb, logic_in_batch_ids, logic_in_ids, logic_types, current_order_mask) 
            var_emb[1, torch.logical_and(torch.tensor(current_order_mask, device=self._device), logic_types==1), :] = negation_out
            conjunction_out = self.conjunction(var_emb, logic_in_batch_ids, logic_in_ids, logic_types, current_order_mask) 
            var_emb[1, torch.logical_and(torch.tensor(current_order_mask, device=self._device), logic_types==2), :] = conjunction_out
            disjunction_out = self.disjunction(var_emb, logic_in_batch_ids, logic_in_ids, logic_types, current_order_mask) 
            var_emb[1, torch.logical_and(torch.tensor(current_order_mask, device=self._device), logic_types==3), :] = disjunction_out

        # return var_emb[1, batch_answer_mask, :]

        # global_seq_embedding = self.node_embedding.weight[input_ids].squeeze()


        # h_masked = self.seq_enc(global_seq_embedding, input_mask, mask_pos)       
        h_masked = var_emb[1, batch_answer_mask>=0, :]
        target_mask_type = mask_type[batch_answer_mask>=0].unsqueeze(-1)
        mask_label = answers[batch_answer_mask>=0]
        batch_size = mask_label.shape[0]
        assert h_masked.shape[0] == batch_size
        assert target_mask_type.shape[0] == batch_size

        prob_out, soft_labels = self.scores_labels(
            h_masked=h_masked, 
            embedding_weight=self.node_embedding.weight, 
            batch_size=batch_size,
            mask_type=target_mask_type, 
            mask_label=mask_label,
            )
        #get loss
        mean_mask_lm_loss = self.myloss(
              logits=prob_out, 
              label=soft_labels,
              )      
        return  mean_mask_lm_loss, prob_out

    def negation(self, var_emb, batch_in_ids, in_ids, types, order_mask):
        negation_basis = torch.ones(size=[in_ids.shape[1], self._emb_size], device=self._device)
        negation_basis = var_emb[0, ...]
        negation_out = self.negation_net(negation_basis)
        negation_mask = torch.logical_and(torch.tensor(order_mask, device=self._device), types==1)
        return (negation_out[negation_mask])

    def conjunction(self, var_emb, batch_in_ids, in_ids, types, order_mask):
        conjunction_basis = torch.ones(size=[in_ids.shape[0], in_ids.shape[1], self._emb_size], device=self._device)
        conjunction_basis[in_ids>=0] = var_emb[1, batch_in_ids[in_ids>=0]]
        conjunction_basis[0, ...] = var_emb[0, ...]
        conjunction_out = self.conjunction_net(conjunction_basis)
        conjunction_mask = torch.logical_and(torch.tensor(order_mask, device=self._device), types==2)
        # return self.act(conjunction_out[conjunction_mask])
        return (conjunction_out[conjunction_mask])

    def disjunction(self, var_emb, batch_in_ids, in_ids, types, order_mask):
        disjunction_basis = torch.zeros(size=[in_ids.shape[0], in_ids.shape[1], self._emb_size], device=self._device)
        disjunction_basis[in_ids>=0] = var_emb[1, batch_in_ids[in_ids>=0]]
        disjunction_basis[0, ...] = var_emb[0, ...]
        disjunction_out = self.disjunction_net(disjunction_basis)
        disjunction_mask = torch.logical_and(torch.tensor(order_mask, device=self._device), types==3)
        return (disjunction_out[disjunction_mask])
