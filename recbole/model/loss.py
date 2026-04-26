# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com


"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding**self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss


class EMOLoss(nn.Module):
    def __init__(self):
        super(EMOLoss, self).__init__()

    @staticmethod
    def multi_hot_embed(masked_index, max_length):
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def forward(self, logits, items, tgt_items, pos_items):
        mle = nn.CrossEntropyLoss()(logits, pos_items)
        mask = self.multi_hot_embed(pos_items, logits.size(1))
        # logits = torch.log(torch.softmax(logits, dim=-1)+1e-8)*mask
        # logits = torch.log(torch.softmax(logits, dim=-1)+1e-8)
        logits = torch.softmax(logits, dim=-1)
        items_normalized = items / torch.norm(items, dim=-1, keepdim=True)
        tgt_item_normalized = tgt_items / torch.norm(tgt_items, dim=-1, keepdim=True)
        weighted_sum = torch.einsum('bi, ij-> bj', logits, items_normalized)
        loss = torch.einsum('bj, bj-> b', weighted_sum, tgt_item_normalized)
        # print(loss.mean())
        # print(loss.size())
        return 1-loss.mean()
        # loss = -0.2*torch.mean(loss) + 0.8*mle


if __name__ == '__main__':
    print(torch.softmax(torch.FloatTensor([0.3, 0.4, 0.3, 0.1, 0.4]), dim=-1))
    print(torch.softmax(torch.FloatTensor([0.3, 0.4, 0.3, 0.1, 0.4]), dim=-1)*torch.FloatTensor([0., 1., 0., 0., 0.]))
