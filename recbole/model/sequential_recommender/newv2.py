import math
import random
import os

import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import networkx as nx
import numpy as np
import torch.nn.functional as F
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class NEWV2(SequentialRecommender):
    def __init__(self, config, dataset):
        super(NEWV2, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.batch_size = config['train_batch_size']
        self.lmd = config['lmd']
        self.tau = config['tau']
        self.sim = config['sim']

        self.disturb = 0.05 if config['disturb'] is None else config['disturb']
        self.stopk = 3 if config['stopk'] is None else config['stopk']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        if dataset.dataset_name == 'ml-100k':
            self.theta = 0.6
        elif dataset.dataset_name == 'beauty':
            self.theta = 0.3
        elif dataset.dataset_name == 'sports':
            self.theta = 0.9
        elif dataset.dataset_name == 'retailrocket-view':
            self.theta = 0.6
        else:
            self.theta = 0.6

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.global_seq1 = nn.Embedding(1, self.hidden_size)
        self.global_seq2 = nn.Embedding(1, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, *args, **kwargs):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()  # [bs, input_len]; mask the 0 item
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_most_sim(self, front, back, front_emb, back_emb):
        sim = torch.einsum('a d, b d -> a b', front_emb, back_emb)
        try:
            top3_indices = torch.topk(sim, k=self.stopk, dim=1, largest=True).indices
            random_mask1 = torch.randint(0, self.stopk, (top3_indices.shape[0],))  # (n,)
            random_mask2 = torch.randint(0, self.stopk, (top3_indices.shape[0],))  # (n,)
        except:
            top3_indices = torch.topk(sim, k=1, dim=1, largest=True).indices
            random_mask1 = torch.randint(0, 1, (top3_indices.shape[0],))  # (n,)
            random_mask2 = torch.randint(0, 1, (top3_indices.shape[0],))  # (n,)
        selected_indices1 = top3_indices[torch.arange(top3_indices.shape[0]), random_mask1]
        selected_indices2 = top3_indices[torch.arange(top3_indices.shape[0]), random_mask2]
        selected_back1 = back[selected_indices1]
        selected_back2 = back[selected_indices2]

        sim = sim.transpose(0, 1)
        try:
            top3_indices = torch.topk(sim, k=self.stopk, dim=1, largest=True).indices
            random_mask1 = torch.randint(0, self.stopk, (top3_indices.shape[0],))  # (n,)
            random_mask2 = torch.randint(0, self.stopk, (top3_indices.shape[0],))
        except:
            top3_indices = torch.topk(sim, k=1, dim=1, largest=True).indices
            random_mask1 = torch.randint(0, 1, (top3_indices.shape[0],))  # (n,)
            random_mask2 = torch.randint(0, 1, (top3_indices.shape[0],))  # (n,)
        selected_indices1 = top3_indices[torch.arange(top3_indices.shape[0]), random_mask1]
        selected_indices2 = top3_indices[torch.arange(top3_indices.shape[0]), random_mask2]
        selected_front1 = front[selected_indices1]
        selected_front2 = front[selected_indices2]
        return selected_back1, selected_back2, selected_front1, selected_front2

    def augment(self, item_seq, item_seq_len, seq_output, int_ent):
        sorted_indices_asc = torch.argsort(int_ent, descending=False)
        item_seq = item_seq[sorted_indices_asc]
        item_seq_len = item_seq_len[sorted_indices_asc]
        seq_output = seq_output[sorted_indices_asc]
        split_idx = int(len(item_seq) * self.disturb)
        front, f_l, f_o = item_seq[:split_idx], item_seq_len[:split_idx], seq_output[:split_idx]
        # m_o = seq_output[split_idx:-split_idx]
        back, b_l, b_o = item_seq[-split_idx:], item_seq_len[-split_idx:], seq_output[-split_idx:]

        b1, b2, f1, f2= self.get_most_sim(front, back, f_o, b_o)
        item_seq1 = torch.cat([b1, front], dim=1)
        item_seq2 = torch.cat([b2, front], dim=1)
        item_seq3 = torch.cat([f1, back], dim=1)
        item_seq4 = torch.cat([f2, back], dim=1)
        mixed1 = torch.cat([item_seq1, torch.cat([torch.zeros_like(item_seq[split_idx:-split_idx]), item_seq[split_idx:-split_idx]], dim=1), item_seq3], dim=0)
        mixed2 = torch.cat([item_seq2, torch.cat([torch.zeros_like(item_seq[split_idx:-split_idx]), item_seq[split_idx:-split_idx]], dim=1), item_seq4], dim=0)
        item_seq_len = item_seq_len + self.max_seq_length
        return mixed1, mixed2, item_seq_len, split_idx

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N - 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

    def forward(self, item_seq, item_seq_len, global_seq, split_idx=None):
        if item_seq.size(1) == 2*self.max_seq_length:
            position_ids = torch.arange(item_seq.size(1) // 2, dtype=torch.long, device=item_seq.device)
            position_ids = torch.cat([position_ids, position_ids], dim=0)
        else:
            position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        if split_idx is not None:
            item_emb[split_idx:-split_idx] += 0.01*torch.randn_like(item_emb[split_idx:-split_idx])
        input_emb = item_emb + position_embedding + global_seq
        # print(global_seq.size(), position_embedding.size(), input_emb.size())
        extended_attention_mask = self.get_attention_mask(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        interest_entropy = interaction['interest_entropy']
        global_seq1 = self.global_seq1(torch.zeros_like(item_seq))
        global_seq2 = self.global_seq2(torch.zeros_like(item_seq))
        global_seq2 = torch.cat([global_seq2, global_seq1], dim=1)
        seq_output = self.forward(item_seq, item_seq_len, global_seq1)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)

        item_seq1, item_seq2, item_seq_len, split_idx = self.augment(item_seq, item_seq_len, seq_output, interest_entropy)
        # split_idx
        seq_output1 = self.forward(item_seq1, item_seq_len, global_seq2, split_idx)
        seq_output2 = self.forward(item_seq2, item_seq_len, global_seq2, split_idx)
        nce_logits, nce_labels = self.info_nce(seq_output1, seq_output2, temp=self.tau, batch_size=seq_output.shape[0],
                                               sim=self.sim, split_idx=split_idx)
        # with torch.no_grad():
        #     s1 = torch.cat([seq_output1[:split_idx], seq_output1[-split_idx:]], dim=0)
        #     s2 = torch.cat([seq_output2[:split_idx], seq_output2[-split_idx:]], dim=0)
        #     s = torch.cat([seq_output[:split_idx], seq_output[-split_idx:]], dim=0)
        #     alignment, uniformity = self.decompose(s1, s2, s,
        #                                            batch_size=s1.shape[0])
        #     directory = '/dev_data/wbq/recbole_iota/recbole_seq/vis/'
        #     log_data = (float(alignment), float(uniformity))

        #     with open(directory + "new_ml100k.txt", "a") as f:
        #         f.write(str(log_data) + "\n")
        nce_loss = self.nce_fct(nce_logits, nce_labels)
        return loss + self.lmd * nce_loss 

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N - 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

    @staticmethod
    def mask_correlated_samples(batch_size):
        """
        correlated sample means the augment samples come from the same naive sample.
        """
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot', split_idx=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N - 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            # embeddings_norm = F.normalize(z, p=2, dim=-1)
            # sim = torch.mm(embeddings_norm, embeddings_norm.transpose(0, 1))
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        # print(sim.size())
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

		# selective alignment
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        positive_samples[split_idx:-split_idx] = 0  # only the first and last split_idx samples are used for alignment

        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        global_seq = self.global_seq1(torch.zeros_like(item_seq))
        seq_output = self.forward(item_seq, item_seq_len, global_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        global_seq = self.global_seq1(torch.zeros_like(item_seq))
        seq_output = self.forward(item_seq, item_seq_len, global_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores