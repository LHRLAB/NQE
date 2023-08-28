from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sre_parse
from sys import stdin
import torch
import torch.nn
from new_model.graph_encoder import encoder,truncated_normal

from loss.loss import softmax_with_cross_entropy
from scoring_function.calc_scores_and_labels import Scores_and_Labels

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

class Sequence_Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Sequence_Encoder,self).__init__()
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._emb_size = config['hidden_size']
        self._intermediate_size = config['intermediate_size']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_dropout_prob']
        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']
        self._n_edge = config['num_edges']
        self._max_seq_len = config['max_seq_len']
        self._max_arity = config['max_arity']
        self._device=config["device"]
        self._L_config = config['L_config']

        self.act = torch.nn.Sigmoid()
        # self.layer_norm1=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True) 
        self.layer_norm1=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=False)

        edge_labels = []
        max_aux = self._max_arity - 2
        max_seq_length = 2 * max_aux + 3
        edge_labels.append([0, 1, 2] + [3,4] * max_aux )
        edge_labels.append([1, 0, 5] + [6,7] * max_aux )
        edge_labels.append([2, 5, 0] + [8,9] * max_aux )
        for idx in range(max_aux):
            edge_labels.append([3,6,8] + [11,12] * idx + [0,10] + [11,12] * (max_aux - idx - 1))
            edge_labels.append([4,7,9] + [12,13] * idx + [10,0] + [12,13] * (max_aux - idx - 1))
        self.edge_labels = torch.tensor(
            data=edge_labels, 
            dtype=torch.int64, 
            device=self._device).unsqueeze(dim=-1)

        #异构图的5种边的参数    [5,H]
        self.edge_embedding_q=torch.nn.Embedding(self._n_edge, self._emb_size // self._n_head)
        self.edge_embedding_q.weight.data=truncated_normal(self.edge_embedding_q.weight.data,std=0.02)
        self.edge_embedding_k=torch.nn.Embedding(self._n_edge, self._emb_size // self._n_head)
        self.edge_embedding_k.weight.data=truncated_normal(self.edge_embedding_k.weight.data,std=0.02)
        self.edge_embedding_v=torch.nn.Embedding(self._n_edge, self._emb_size // self._n_head)
        self.edge_embedding_v.weight.data=truncated_normal(self.edge_embedding_v.weight.data,std=0.02)
        #编码器
        self.encoder_model=encoder( 
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._intermediate_size,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            device=self._device,
            L_config=self._L_config)

        # One linear layer
        self.fc1=torch.nn.Linear(self._emb_size, self._emb_size)
        self.fc1.weight.data=truncated_normal(self.fc1.weight.data,std=0.02)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        self.layer_norm2=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True)
        
        
    def forward(self, global_seq_embedding, input_mask, mask_pos):

        emb_out = self.act(torch.nn.Dropout(self._prepostprocess_dropout)(self.layer_norm1(global_seq_embedding)))

        # get edge embeddings between input tokens
        edges_query = self.edge_embedding_q(torch.squeeze(self.edge_labels))
        edges_key = self.edge_embedding_k(torch.squeeze(self.edge_labels))
        edges_value = self.edge_embedding_v(torch.squeeze(self.edge_labels))
        edge_mask = torch.sign(self.edge_labels) 
        edges_query = torch.mul(edges_query, edge_mask)
        edges_key = torch.mul(edges_key, edge_mask)
        edges_value = torch.mul(edges_value, edge_mask)
        # get multi-head self-attention mask
        self_attn_mask = torch.matmul(input_mask,input_mask.transpose(1,2))
        self_attn_mask=1000000.0*(self_attn_mask-1.0)
        n_head_self_attn_mask = torch.stack([self_attn_mask] * self._n_head, dim=1)###1024x4个相同的11x64个mask
        # stack of graph transformer encoders       
        _enc_out = self.encoder_model(
            enc_input=emb_out,
            edges_query=edges_query,
            edges_key=edges_key,
            edges_value=edges_value,
            attn_bias=n_head_self_attn_mask)   

        h_masked = _enc_out.gather(dim=1, index=mask_pos.unsqueeze(-1).repeat(1, 1, self._emb_size)).squeeze()
        # transform: fc1
        h_masked=torch.nn.GELU()(self.fc1(h_masked))
        # transform: layer norm
        h_masked=self.layer_norm2(h_masked)

        return h_masked  





        
