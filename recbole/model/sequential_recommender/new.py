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


class EGPF(nn.Module):
    def __init__(self, beta=0.5, gamma=0.7, lambda_proto=0.1, eps=1e-12):
        super(EGPF, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.lambda_proto = lambda_proto
        self.eps = eps

    def forward(self, z_enc, prototypes, proto_weights, entropy, is_medium):
        """
        EGPF 仅对中熵序列生效，低/高熵序列保持原编码表示不变。

        公式实现：
        1) tau(H(s)) = beta * H(s)
        2) alpha_j = softmax(w_j / tau(H(s)))
        3) z_proto = sum_j alpha_j * p_j
        4) z_s = gamma * z_enc + (1-gamma) * z_proto
        5) L_proto = lambda_proto * ||z_s - z_proto||_2^2（仅中熵）
        """
        medium_mask = is_medium.bool()

        # 对应公式 tau(H(s)) = beta * H(s)，并做数值稳定保护
        tau_h = torch.clamp(self.beta * entropy, min=self.eps).unsqueeze(-1)

        # 对齐公式 alpha_j = exp(w_j / tau) / sum_k exp(w_k / tau)
        logits = proto_weights / tau_h
        valid_mask = proto_weights > 0
        logits = logits.masked_fill(~valid_mask, -1e9)
        alpha = torch.softmax(logits, dim=-1)

        # 防止填充位参与加权，重新归一化
        alpha = alpha * valid_mask.float()
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # 对应公式 z_proto = sum_j alpha_j * p_j
        z_proto = torch.sum(alpha.unsqueeze(-1) * prototypes, dim=1)

        # 默认输出为原编码，确保 low/high entropy 完全保持原流程
        z_s = z_enc
        if medium_mask.any():
            # 对应公式 z_s = gamma * z_enc + (1-gamma) * z_proto
            z_s = z_enc.clone()
            z_s[medium_mask] = (
                self.gamma * z_enc[medium_mask]
                + (1.0 - self.gamma) * z_proto[medium_mask]
            )
            # 对应公式 L_proto = lambda_proto * ||z_s - z_proto||_2^2（仅中熵样本）
            proto_dist = torch.sum((z_s[medium_mask] - z_proto[medium_mask]) ** 2, dim=-1)
            proto_loss = self.lambda_proto * proto_dist.mean()
        else:
            proto_loss = z_enc.new_zeros(())

        return z_s, proto_loss


class NEW(SequentialRecommender):
    def __init__(self, config, dataset):
        super(NEW, self).__init__(config, dataset)

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

        # EGPF 超参数（均可通过配置文件覆盖）
        self.egpf_beta = config['egpf_beta'] if 'egpf_beta' in config else 0.5
        self.egpf_gamma = config['egpf_gamma'] if 'egpf_gamma' in config else 0.7
        self.lambda_proto = config['lambda_proto'] if 'lambda_proto' in config else 0.1
        self.egpf_warmup_epochs = config['egpf_warmup_epochs'] if 'egpf_warmup_epochs' in config else 20
        self.egpf_apply_interval = config['egpf_apply_interval'] if 'egpf_apply_interval' in config else 1
        self.egpf_quantile_low = config['egpf_quantile_low'] if 'egpf_quantile_low' in config else 0.3
        self.egpf_quantile_high = config['egpf_quantile_high'] if 'egpf_quantile_high' in config else 0.7
        self.egpf_use_running_quantile = config['egpf_use_running_quantile'] if 'egpf_use_running_quantile' in config else True
        self.egpf_quantile_momentum = config['egpf_quantile_momentum'] if 'egpf_quantile_momentum' in config else 0.9

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
        self.egpf = EGPF(
            beta=self.egpf_beta,
            gamma=self.egpf_gamma,
            lambda_proto=self.lambda_proto,
        )
        self.current_epoch = 0
        self.train_step = 0
        self.running_low_thresh = None
        self.running_high_thresh = None

        # parameters initialization
        self.apply(self._init_weights)

    def set_train_epoch(self, epoch_idx):
        self.current_epoch = epoch_idx

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

    def get_connected_components_sizes(self, batch_adj, item_seq_len):
        if isinstance(batch_adj, torch.Tensor):
            batch_adj = batch_adj.detach().cpu().numpy()
        
        batch_size = batch_adj.shape[0]
        all_sizes = []
        
        for i in range(batch_size):
            adj = batch_adj[i]
            seq_len = item_seq_len[i]
            G = nx.from_numpy_array(adj[:seq_len, :seq_len])
            components = list(nx.connected_components(G))
            sizes = [len(c) for c in components]
            all_sizes.append(sizes)
        return all_sizes

    def interest_entropy(self, item_seq, item_seq_len, theta=0.9):
        entropy, _, _, _ = self.build_entropy_prototypes(item_seq, item_seq_len, theta)
        return entropy

    def build_entropy_prototypes(self, item_seq, item_seq_len, theta=0.9):
        """
        在当前 batch 上同时构建：
        - entropy: 每条序列的兴趣熵 H(s)
        - prototypes: 每条序列的兴趣原型矩阵 [B, m, H]（按 batch 内最大 m 做零填充）
        - proto_weights: 每个原型权重（连通分量大小归一化）[B, m]
        - is_medium: 基于 τ=0.3 分位划分的中熵掩码
        """
        embeddings = self.item_embedding(item_seq)
        batch_size, num_items, _ = embeddings.shape

        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.bmm(embeddings_norm, embeddings_norm.transpose(1, 2))
        adj_matrix = (sim_matrix > theta).float()

        rows = torch.arange(num_items, device=self.device).view(1, num_items, 1)  #
        cols = torch.arange(num_items, device=self.device).view(1, 1, num_items)  #
        mask = (rows >= item_seq_len.view(batch_size, 1, 1)) & (cols >= item_seq_len.view(batch_size, 1, 1))  #
        adj_matrix[mask] = 0

        adj_np = adj_matrix.detach().cpu().numpy()
        seq_lens = item_seq_len.detach().cpu().tolist()

        entropy_list = []
        prototype_list = []
        weight_list = []

        for idx, seq_len in enumerate(seq_lens):
            seq_len = int(seq_len)
            if seq_len <= 0:
                entropy_list.append(torch.tensor(0.0, device=item_seq.device, dtype=embeddings.dtype))
                prototype_list.append(torch.zeros((1, self.hidden_size), device=item_seq.device, dtype=embeddings.dtype))
                weight_list.append(torch.ones((1,), device=item_seq.device, dtype=embeddings.dtype))
                continue

            graph = nx.from_numpy_array(adj_np[idx][:seq_len, :seq_len])
            components = list(nx.connected_components(graph))

            comp_sizes = [len(component) for component in components]
            comp_weights = torch.tensor(
                [size / float(seq_len) for size in comp_sizes],
                device=item_seq.device,
                dtype=embeddings.dtype,
            )
            entropy = -torch.sum(comp_weights * torch.log2(comp_weights + 1e-10))

            sample_prototypes = []
            for component in components:
                comp_idx = torch.tensor(list(component), device=item_seq.device, dtype=torch.long)
                comp_proto = embeddings[idx, comp_idx].mean(dim=0)
                sample_prototypes.append(comp_proto)

            sample_prototypes = torch.stack(sample_prototypes, dim=0)
            entropy_list.append(entropy)
            prototype_list.append(sample_prototypes)
            weight_list.append(comp_weights)

        entropy = torch.stack(entropy_list)
        max_m = max(proto.size(0) for proto in prototype_list)

        prototypes = torch.zeros(
            (batch_size, max_m, self.hidden_size),
            device=item_seq.device,
            dtype=embeddings.dtype,
        )
        proto_weights = torch.zeros(
            (batch_size, max_m),
            device=item_seq.device,
            dtype=embeddings.dtype,
        )

        for idx in range(batch_size):
            m = prototype_list[idx].size(0)
            prototypes[idx, :m] = prototype_list[idx]
            proto_weights[idx, :m] = weight_list[idx]

        # 分位划分：默认 q30/q70；可选使用平滑阈值降低 batch 抖动
        batch_low = torch.quantile(entropy, self.egpf_quantile_low)
        batch_high = torch.quantile(entropy, self.egpf_quantile_high)

        if self.egpf_use_running_quantile:
            if self.running_low_thresh is None or self.running_high_thresh is None:
                self.running_low_thresh = float(batch_low.item())
                self.running_high_thresh = float(batch_high.item())
            elif self.training:
                momentum = self.egpf_quantile_momentum
                self.running_low_thresh = momentum * self.running_low_thresh + (1.0 - momentum) * float(batch_low.item())
                self.running_high_thresh = momentum * self.running_high_thresh + (1.0 - momentum) * float(batch_high.item())

            low_thresh = entropy.new_tensor(self.running_low_thresh)
            high_thresh = entropy.new_tensor(self.running_high_thresh)
        else:
            low_thresh = batch_low
            high_thresh = batch_high

        is_medium = (entropy >= low_thresh) & (entropy <= high_thresh)

        return entropy, prototypes, proto_weights, is_medium

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

    def augment(self, item_seq, item_seq_len, seq_output, int_ent=None):
        if int_ent is None:
            int_ent = self.interest_entropy(item_seq, item_seq_len, theta=self.theta)
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

    def forward(
        self,
        item_seq,
        item_seq_len,
        global_seq,
        entropy=None,
        prototypes=None,
        proto_weights=None,
        is_medium=None,
        return_proto_loss=False,
    ):
        if item_seq.size(1) == 2*self.max_seq_length:
            position_ids = torch.arange(item_seq.size(1) // 2, dtype=torch.long, device=item_seq.device)
            position_ids = torch.cat([position_ids, position_ids], dim=0)
        else:
            position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding + global_seq
        # print(global_seq.size(), position_embedding.size(), input_emb.size())
        extended_attention_mask = self.get_attention_mask(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        z_enc = self.gather_indexes(output, item_seq_len - 1)

        proto_loss = z_enc.new_zeros(())
        z_s = z_enc
        # 仅当中熵掩码及原型信息可用时，执行 EGPF 融合
        if (
            entropy is not None
            and prototypes is not None
            and proto_weights is not None
            and is_medium is not None
        ):
            z_s, proto_loss = self.egpf(
                z_enc=z_enc,
                prototypes=prototypes,
                proto_weights=proto_weights,
                entropy=entropy,
                is_medium=is_medium,
            )

        if return_proto_loss:
            return z_s, proto_loss
        return z_s  # [B H]
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        global_seq1 = self.global_seq1(torch.zeros_like(item_seq))
        global_seq2 = self.global_seq2(torch.zeros_like(item_seq))
        global_seq2 = torch.cat([global_seq2, global_seq1], dim=1)
        self.train_step += 1

        # EGPF 加速策略：warmup 期关闭；并支持按间隔启用以减少每个 epoch 计算开销
        apply_egpf = (
            self.current_epoch >= self.egpf_warmup_epochs
            and self.egpf_apply_interval > 0
            and self.train_step % self.egpf_apply_interval == 0
        )

        if apply_egpf:
            entropy, prototypes, proto_weights, is_medium = self.build_entropy_prototypes(
                item_seq, item_seq_len, theta=self.theta
            )
            seq_output, proto_loss = self.forward(
                item_seq,
                item_seq_len,
                global_seq1,
                entropy=entropy,
                prototypes=prototypes,
                proto_weights=proto_weights,
                is_medium=is_medium,
                return_proto_loss=True,
            )
        else:
            entropy = self.interest_entropy(item_seq, item_seq_len, theta=self.theta)
            seq_output = self.forward(item_seq, item_seq_len, global_seq1)
            proto_loss = seq_output.new_zeros(())

        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)

        # 保持原 SLA + SA-CL 流程不变（仅使用原 IERec 的增强策略）
        item_seq1, item_seq2, item_seq_len, split_idx = self.augment(item_seq, item_seq_len, seq_output, int_ent=entropy)
        seq_output1 = self.forward(item_seq1, item_seq_len, global_seq2)
        seq_output2 = self.forward(item_seq2, item_seq_len, global_seq2)
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
        return loss + self.lmd * nce_loss + proto_loss

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
        entropy, prototypes, proto_weights, is_medium = self.build_entropy_prototypes(
            item_seq, item_seq_len, theta=self.theta
        )
        seq_output = self.forward(
            item_seq,
            item_seq_len,
            global_seq,
            entropy=entropy,
            prototypes=prototypes,
            proto_weights=proto_weights,
            is_medium=is_medium,
        )
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        global_seq = self.global_seq1(torch.zeros_like(item_seq))
        entropy, prototypes, proto_weights, is_medium = self.build_entropy_prototypes(
            item_seq, item_seq_len, theta=self.theta
        )
        seq_output = self.forward(
            item_seq,
            item_seq_len,
            global_seq,
            entropy=entropy,
            prototypes=prototypes,
            proto_weights=proto_weights,
            is_medium=is_medium,
        )
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores