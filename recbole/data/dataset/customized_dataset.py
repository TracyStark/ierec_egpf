# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""

import numpy as np
import torch

from recbole.data.dataset import KGSeqDataset, SequentialDataset
from recbole.data.interaction import Interaction
from recbole.sampler import SeqSampler
from recbole.utils.enum_type import FeatureType

# DCRec
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
from recbole.data import (
    create_dataset,
    data_preparation,
    save_split_dataloaders,
    load_split_dataloaders,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
import torch.nn.functional as F
import networkx as nx
# import dgl

class GRU4RecKGDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class KSRDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class DIENDataset(SequentialDataset):
    """:class:`DIENDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It add users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored DIENDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def __init__(self, config):
        super().__init__(config)

        list_suffix = config["LIST_SUFFIX"]
        neg_prefix = config["NEG_PREFIX"]
        self.seq_sampler = SeqSampler(self)
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix
        self.neg_item_list = self.seq_sampler.sample_neg_sequence(
            self.inter_feat[self.iid_field]
        )

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                list_ftype = self.field2type[list_field]
                dtype = (
                    torch.int64
                    if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]
                    else torch.float64
                )
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

                # DIEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    for i, (index, length) in enumerate(
                        zip(item_list_index, item_list_length)
                    ):
                        new_dict[self.neg_item_list_field][i][
                            :length
                        ] = self.neg_item_list[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data


import numpy as np
import torch

from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType, FeatureSource


class DCRecDataset(Dataset):
    """:class:`SequentialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and provides augmentation interface to adapt to Sequential Recommendation,
    which can accelerate the data loader.

    Attributes:
        max_item_list_len (int): Max length of historical item list.
        item_list_length_field (str): Field name for item lists' length.
    """

    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        self.sim_group = config['sim_group']
        self.external_data = {
            "adj_graph": None,
            "sim_graph": None,
            "user_edges": None,
            "adj_graph_val": None,
            "sim_graph_val": None,
            "adj_graph_test": None,
            "sim_graph_test": None
        }
        super().__init__(config)
        if config["benchmark_filename"] is not None:
            self._benchmark_presets()

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        super()._change_feat_format()

        if self.config["benchmark_filename"] is not None:
            return
        self.logger.debug("Augmentation for sequential recommendation.")
        self.data_augmentation()

    def _aug_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(
                    list_field, list_ftype, FeatureSource.INTERACTION, list_len
                )

        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                new_dict[list_field] = torch.zeros(
                    shape, dtype=self.inter_feat[field].dtype
                )

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

    def _benchmark_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )
        self.inter_feat[self.item_list_length_field] = self.inter_feat[
            self.item_id_list_field
        ].agg(len)

    def inter_matrix(self, form="coo", value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                "dataset does not exist uid/iid, thus can not converted to sparse matrix."
            )

        l1_idx = self.inter_feat[self.item_list_length_field] == 1
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config["LIST_SUFFIX"]
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]]
                )
            elif (not field.endswith(list_suffix)) and (
                field != self.item_list_length_field
            ):
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field]]
                )
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(
            local_inter_feat, self.uid_field, self.iid_field, form, value_field
        )

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of built :class:`Dataset`.
        """
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )
        trn, val, tst = super().build()
        adj_graph, user_edges = self.build_adj_graph(trn)
        adj_graph_val, _ = self.build_adj_graph(val, "val")
        adj_graph_test, _ = self.build_adj_graph(tst, "test")
        sim_graph = self.build_sim_graph(trn, self.sim_group)
        sim_graph_val = self.build_sim_graph(val, self.sim_group, "val")
        sim_graph_test = self.build_sim_graph(tst, self.sim_group, "test")
        external_data = {
            "adj_graph": adj_graph,
            "sim_graph": sim_graph,
            "user_edges": user_edges,
            "adj_graph_val": adj_graph_val,
            "sim_graph_val": sim_graph_val,
            "adj_graph_test": adj_graph_test,
            "sim_graph_test": sim_graph_test
        }
        trn.external_data = external_data
        val.external_data = external_data
        tst.external_data = external_data
        return trn, val, tst

    def build_adj_graph(self, dataset, phase="train"):
        print("constructing DGL graph...")
        item_adj_dict = defaultdict(list)
        item_edges_of_user = dict()
        inter_feat = dataset.inter_feat
        old_user = -1
        mem_item_seq = []
        for line in range(len(inter_feat)):
            item_edges_a, item_edges_b = [], []
            uid = inter_feat[dataset.uid_field][line].item()
            item_seq = inter_feat[dataset.item_id_list_field][line].tolist()
            seq_len = inter_feat[dataset.item_list_length_field][line].item()
            item_seq = item_seq[:seq_len]
            if old_user == -1:
                old_user = uid
                mem_item_seq.append(item_seq[-1])
                continue
            if uid == old_user:
                mem_item_seq.append(item_seq[-1])
                continue
            for i in range(len(mem_item_seq)):
                if i > 0:
                    item_adj_dict[mem_item_seq[i]].append(mem_item_seq[i-1])
                    item_adj_dict[mem_item_seq[i-1]].append(mem_item_seq[i])
                    item_edges_a.append(mem_item_seq[i])
                    item_edges_b.append(mem_item_seq[i-1])
                if i+1 < len(mem_item_seq):
                    item_adj_dict[mem_item_seq[i]].append(mem_item_seq[i+1])
                    item_adj_dict[mem_item_seq[i+1]].append(mem_item_seq[i])
                    item_edges_a.append(mem_item_seq[i])
                    item_edges_b.append(mem_item_seq[i+1])
            item_edges_of_user[old_user] = (np.asarray(item_edges_a, dtype=np.int64), np.asarray(item_edges_b, dtype=np.int64))
            old_user = uid
            mem_item_seq = [item_seq[-1]]
        item_edges_of_user = pd.DataFrame.from_dict(item_edges_of_user, orient='index', columns=['item_edges_a', 'item_edges_b'])
        # item_edges_of_user.to_pickle(user_edges_file)
        cols = []
        rows = []
        values = []
        for item in item_adj_dict:
            adj = item_adj_dict[item]
            adj_count = Counter(adj)

            rows.extend([item]*len(adj_count))
            cols.extend(adj_count.keys())
            values.extend(adj_count.values())

        adj_mat = csr_matrix((values, (rows, cols)), shape=(
            dataset.item_num + 1, dataset.item_num + 1))
        adj_mat = adj_mat.tolil()
        adj_mat.setdiag(np.ones((dataset.item_num + 1,)))
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        g = dgl.from_scipy(norm_adj, 'w', idtype=torch.int64)
        g.edata['w'] = g.edata['w'].float()
        # print("saving DGL graph to binary file...")
        # dgl.save_graphs(graph_file, [g])
        return g, item_edges_of_user
    
    def build_sim_graph(self, dataset, k, phase="train"):
        import dgl
        # graph_file = dataset.config['data_path']+f"/sim_graph_g{k}_{phase}.bin"
        # try:
        #     g = dgl.load_graphs(graph_file, [0])
        #     print("loading isim graph from DGL binary file...")
        #     return g[0][0]
        # except:
        print("building isim graph...")
        row = []
        col = []
        inter_feat = dataset.inter_feat
        old_user = -1
        mem_item_seq = []
        for line in range(len(dataset.inter_feat)):
            uid = inter_feat[dataset.uid_field][line].item()
            item_seq = inter_feat[dataset.item_id_list_field][line].tolist()
            seq_len = inter_feat[dataset.item_list_length_field][line].item()
            item_seq = item_seq[:seq_len]
            if old_user == -1:
                old_user = uid
                mem_item_seq.append(item_seq[-1])
                continue
            if uid == old_user:
                mem_item_seq.append(item_seq[-1])
                continue
            col.extend(mem_item_seq)
            row.extend([uid]*len(mem_item_seq))
            old_user = uid
            mem_item_seq = [item_seq[-1]]

        row = np.array(row)
        col = np.array(col)
        # n_users, n_items
        cf_graph = csr_matrix(([1]*len(row), (row, col)), shape=(
            dataset.user_num+1, dataset.item_num+1), dtype=np.float32)
        similarity = cosine_similarity(cf_graph.transpose())
        # filter topk connections
        sim_items_slices = []
        sim_weights_slices = []
        i = 0
        while i < similarity.shape[0]:
            similarity = similarity[i:, :]
            sim = similarity[:256, :]
            sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
            sim_weights = np.take_along_axis(sim, sim_items, axis=1)
            sim_items_slices.append(sim_items)
            sim_weights_slices.append(sim_weights)
            i = i + 256
        sim = similarity[256:, :]
        sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
        sim_weights = np.take_along_axis(sim, sim_items, axis=1)
        sim_items_slices.append(sim_items)
        sim_weights_slices.append(sim_weights)

        sim_items = np.concatenate(sim_items_slices, axis=0)
        sim_weights = np.concatenate(sim_weights_slices, axis=0)
        row = []
        col = []
        for i in range(len(sim_items)):
            row.extend([i]*len(sim_items[i]))
            col.extend(sim_items[i])
        values = sim_weights / (sim_weights.sum(axis=1, keepdims=True)+1e-24)
        values = np.nan_to_num(values).flatten()
        adj_mat = csr_matrix((values, (row, col)), shape=(
            dataset.item_num + 1, dataset.item_num + 1))
        g = dgl.from_scipy(adj_mat, 'w')
        g.edata['w'] = g.edata['w'].float()
        # print("saving isim graph to binary file...")
        # dgl.save_graphs(graph_file, [g])
        return g
    
class DuoRecDataset(Dataset):
    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        super().__init__(config)
        if config["benchmark_filename"] is not None:
            self._benchmark_presets()
    
    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        super()._change_feat_format()

        if self.config["benchmark_filename"] is not None:
            return
        self.logger.debug("Augmentation for sequential recommendation.")
        self.data_augmentation()

    def _aug_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(
                    list_field, list_ftype, FeatureSource.INTERACTION, list_len
                )

        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )

    def data_augmentation(self):
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                new_dict[list_field] = torch.zeros(
                    shape, dtype=self.inter_feat[field].dtype
                )

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

    def semantic_augmentation(self, target_item):
        same_target_index = []
        # target_item = self.inter_feat['item_id'][target_index].numpy()
        for index, item_id in enumerate(target_item):
            all_index_same_id = np.where(target_item == item_id)[0]  # all index of a specific item id with self item
            delete_index = np.argwhere(all_index_same_id == index)
            all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)
            same_target_index.append(all_index_same_id_wo_self)
        # same_target_index = np.array(same_target_index)
        # np.save(aug_path, same_target_index)
        return same_target_index

    def inter_matrix(self, form="coo", value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                "dataset does not exist uid/iid, thus can not converted to sparse matrix."
            )

        l1_idx = self.inter_feat[self.item_list_length_field] == 1
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config["LIST_SUFFIX"]
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]]
                )
            elif (not field.endswith(list_suffix)) and (
                field != self.item_list_length_field
            ):
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field]]
                )
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(
            local_inter_feat, self.uid_field, self.iid_field, form, value_field
        )

    def build(self):
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )
        trn, val, tst = super().build()
        inter_feat = trn.inter_feat
        same_target_index = self.semantic_augmentation(inter_feat[trn.iid_field])
        new_dict = {
            'sem_aug': None,
            'sem_aug_lengths': None,
        }
        sample_pos = []
        null_index = []
        for i, targets in enumerate(same_target_index):
            if len(targets) == 0:
                sample_pos.append(-1)
                null_index.append(i)
            else:
                sample_pos.append(np.random.choice(targets))
        sem_pos_seqs = inter_feat[trn.item_id_list_field][sample_pos]
        sem_pos_lengths = inter_feat[trn.item_list_length_field][sample_pos]
        if null_index:
            sem_pos_seqs[null_index] = inter_feat[trn.item_id_list_field][null_index]
            sem_pos_lengths[null_index] = inter_feat[trn.item_list_length_field][null_index]
        new_dict["sem_aug"] = sem_pos_seqs
        new_dict["sem_aug_lengths"] = sem_pos_lengths
        inter_feat.update(Interaction(new_dict))
        return trn, val, tst


class DuoRecUniDataset(Dataset):
    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        super().__init__(config)
        if config["benchmark_filename"] is not None:
            self._benchmark_presets()
    
    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        super()._change_feat_format()

        if self.config["benchmark_filename"] is not None:
            return
        self.logger.debug("Augmentation for sequential recommendation.")
        self.data_augmentation()

    def _aug_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(
                    list_field, list_ftype, FeatureSource.INTERACTION, list_len
                )

        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )

    def data_augmentation(self):
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                new_dict[list_field] = torch.zeros(
                    shape, dtype=self.inter_feat[field].dtype
                )

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

    def semantic_augmentation(self, target_item):
        same_target_index = []
        # target_item = self.inter_feat['item_id'][target_index].numpy()
        for index, item_id in enumerate(target_item):
            all_index_same_id = np.where(target_item == item_id)[0]  # all index of a specific item id with self item
            delete_index = np.argwhere(all_index_same_id == index)
            all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)
            same_target_index.append(all_index_same_id_wo_self)
        # same_target_index = np.array(same_target_index)
        # np.save(aug_path, same_target_index)
        return same_target_index

    def inter_matrix(self, form="coo", value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                "dataset does not exist uid/iid, thus can not converted to sparse matrix."
            )

        l1_idx = self.inter_feat[self.item_list_length_field] == 1
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config["LIST_SUFFIX"]
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]]
                )
            elif (not field.endswith(list_suffix)) and (
                field != self.item_list_length_field
            ):
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field]]
                )
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(
            local_inter_feat, self.uid_field, self.iid_field, form, value_field
        )

    def build(self):
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )
        trn, val, tst = super().build()
        inter_feat = trn.inter_feat
        same_target_index = self.semantic_augmentation(inter_feat[trn.iid_field])
        new_dict = {
            'sem_aug': None,
            'sem_aug_lengths': None,
        }
        sample_pos = []
        null_index = []
        for i, targets in enumerate(same_target_index):
            if len(targets) == 0:
                sample_pos.append(-1)
                null_index.append(i)
            else:
                sample_pos.append(np.random.choice(targets))
        sem_pos_seqs = inter_feat[trn.item_id_list_field][sample_pos]
        sem_pos_lengths = inter_feat[trn.item_list_length_field][sample_pos]
        if null_index:
            sem_pos_seqs[null_index] = inter_feat[trn.item_id_list_field][null_index]
            sem_pos_lengths[null_index] = inter_feat[trn.item_list_length_field][null_index]
        new_dict["sem_aug"] = sem_pos_seqs
        new_dict["sem_aug_lengths"] = sem_pos_lengths
        inter_feat.update(Interaction(new_dict))
        return trn, val, tst

class DuoRecSLADataset(Dataset):
    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        super().__init__(config)
        if config["benchmark_filename"] is not None:
            self._benchmark_presets()
    
    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        super()._change_feat_format()

        if self.config["benchmark_filename"] is not None:
            return
        self.logger.debug("Augmentation for sequential recommendation.")
        self.data_augmentation()

    def _aug_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(
                    list_field, list_ftype, FeatureSource.INTERACTION, list_len
                )

        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )

    def data_augmentation(self):
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                new_dict[list_field] = torch.zeros(
                    shape, dtype=self.inter_feat[field].dtype
                )

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

    def semantic_augmentation(self, target_item):
        same_target_index = []
        # target_item = self.inter_feat['item_id'][target_index].numpy()
        for index, item_id in enumerate(target_item):
            all_index_same_id = np.where(target_item == item_id)[0]  # all index of a specific item id with self item
            delete_index = np.argwhere(all_index_same_id == index)
            all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)
            same_target_index.append(all_index_same_id_wo_self)
        # same_target_index = np.array(same_target_index)
        # np.save(aug_path, same_target_index)
        return same_target_index

    def inter_matrix(self, form="coo", value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                "dataset does not exist uid/iid, thus can not converted to sparse matrix."
            )

        l1_idx = self.inter_feat[self.item_list_length_field] == 1
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config["LIST_SUFFIX"]
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]]
                )
            elif (not field.endswith(list_suffix)) and (
                field != self.item_list_length_field
            ):
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field]]
                )
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(
            local_inter_feat, self.uid_field, self.iid_field, form, value_field
        )

    def build(self):
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )
        trn, val, tst = super().build()
        inter_feat = trn.inter_feat
        same_target_index = self.semantic_augmentation(inter_feat[trn.iid_field])
        new_dict = {
            'sem_aug': None,
            'sem_aug_lengths': None,
        }
        sample_pos = []
        null_index = []
        for i, targets in enumerate(same_target_index):
            if len(targets) == 0:
                sample_pos.append(-1)
                null_index.append(i)
            else:
                sample_pos.append(np.random.choice(targets))
        sem_pos_seqs = inter_feat[trn.item_id_list_field][sample_pos]
        sem_pos_lengths = inter_feat[trn.item_list_length_field][sample_pos]
        if null_index:
            sem_pos_seqs[null_index] = inter_feat[trn.item_id_list_field][null_index]
            sem_pos_lengths[null_index] = inter_feat[trn.item_list_length_field][null_index]
        new_dict["sem_aug"] = sem_pos_seqs
        new_dict["sem_aug_lengths"] = sem_pos_lengths
        inter_feat.update(Interaction(new_dict))
        return trn, val, tst


class MAERecDataset(Dataset):
    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        self.ii_dok = None
        self.ii_adj = None
        self.ii_adj_all_one = None
        self.sim_group = config['sim_group']
        super().__init__(config)

    def _change_feat_format(self):
        super()._change_feat_format()

        if self.config["benchmark_filename"] is not None:
            return
        self.logger.debug("Augmentation for sequential recommendation.")
        self.data_augmentation()

    def _aug_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(
                    list_field, list_ftype, FeatureSource.INTERACTION, list_len
                )

        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )

    def data_augmentation(self):
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                new_dict[list_field] = torch.zeros(
                    shape, dtype=self.inter_feat[field].dtype
                )

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data


    def build(self):
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )
        trn, val, tst = super().build()
        adj_graph, user_edges = self.build_adj_graph(trn)
        adj_graph_val, _ = self.build_adj_graph(val, "val")
        adj_graph_test, _ = self.build_adj_graph(tst, "test")
        sim_graph = self.build_sim_graph(trn, self.sim_group)
        sim_graph_val = self.build_sim_graph(val, self.sim_group, "val")
        sim_graph_test = self.build_sim_graph(tst, self.sim_group, "test")
        external_data = {
            "adj_graph": adj_graph,
            "sim_graph": sim_graph,
            "user_edges": user_edges,
            "adj_graph_val": adj_graph_val,
            "sim_graph_val": sim_graph_val,
            "adj_graph_test": adj_graph_test,
            "sim_graph_test": sim_graph_test
        }
        trn.external_data = external_data
        val.external_data = external_data
        tst.external_data = external_data
        return trn, val, tst

    def build_adj_graph(self, dataset, phase="train"):
        print("constructing DGL graph...")
        item_adj_dict = defaultdict(list)
        item_edges_of_user = dict()
        inter_feat = dataset.inter_feat
        old_user = -1
        mem_item_seq = []
        for line in range(len(inter_feat)):
            item_edges_a, item_edges_b = [], []
            uid = inter_feat[dataset.uid_field][line].item()
            item_seq = inter_feat[dataset.item_id_list_field][line].tolist()
            seq_len = inter_feat[dataset.item_list_length_field][line].item()
            item_seq = item_seq[:seq_len]
            if old_user == -1:
                old_user = uid
                mem_item_seq.append(item_seq[-1])
                continue
            if uid == old_user:
                mem_item_seq.append(item_seq[-1])
                continue
            for i in range(len(mem_item_seq)):
                if i > 0:
                    item_adj_dict[mem_item_seq[i]].append(mem_item_seq[i-1])
                    item_adj_dict[mem_item_seq[i-1]].append(mem_item_seq[i])
                    item_edges_a.append(mem_item_seq[i])
                    item_edges_b.append(mem_item_seq[i-1])
                if i+1 < len(mem_item_seq):
                    item_adj_dict[mem_item_seq[i]].append(mem_item_seq[i+1])
                    item_adj_dict[mem_item_seq[i+1]].append(mem_item_seq[i])
                    item_edges_a.append(mem_item_seq[i])
                    item_edges_b.append(mem_item_seq[i+1])
            item_edges_of_user[old_user] = (np.asarray(item_edges_a, dtype=np.int64), np.asarray(item_edges_b, dtype=np.int64))
            old_user = uid
            mem_item_seq = [item_seq[-1]]
        item_edges_of_user = pd.DataFrame.from_dict(item_edges_of_user, orient='index', columns=['item_edges_a', 'item_edges_b'])
        # item_edges_of_user.to_pickle(user_edges_file)
        cols = []
        rows = []
        values = []
        for item in item_adj_dict:
            adj = item_adj_dict[item]
            adj_count = Counter(adj)

            rows.extend([item]*len(adj_count))
            cols.extend(adj_count.keys())
            values.extend(adj_count.values())

        adj_mat = csr_matrix((values, (rows, cols)), shape=(
            dataset.item_num + 1, dataset.item_num + 1))
        adj_mat = adj_mat.tolil()
        adj_mat.setdiag(np.ones((dataset.item_num + 1,)))
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        g = dgl.from_scipy(norm_adj, 'w', idtype=torch.int64)
        g.edata['w'] = g.edata['w'].float()
        # print("saving DGL graph to binary file...")
        # dgl.save_graphs(graph_file, [g])
        return g, item_edges_of_user
    
    def build_sim_graph(self, dataset, k, phase="train"):
        import dgl
        print("building isim graph...")
        row = []
        col = []
        inter_feat = dataset.inter_feat
        old_user = -1
        mem_item_seq = []
        for line in range(len(dataset.inter_feat)):
            uid = inter_feat[dataset.uid_field][line].item()
            item_seq = inter_feat[dataset.item_id_list_field][line].tolist()
            seq_len = inter_feat[dataset.item_list_length_field][line].item()
            item_seq = item_seq[:seq_len]
            if old_user == -1:
                old_user = uid
                mem_item_seq.append(item_seq[-1])
                continue
            if uid == old_user:
                mem_item_seq.append(item_seq[-1])
                continue
            col.extend(mem_item_seq)
            row.extend([uid]*len(mem_item_seq))
            old_user = uid
            mem_item_seq = [item_seq[-1]]

        row = np.array(row)
        col = np.array(col)
        # n_users, n_items
        cf_graph = csr_matrix(([1]*len(row), (row, col)), shape=(
            dataset.user_num+1, dataset.item_num+1), dtype=np.float32)
        similarity = cosine_similarity(cf_graph.transpose())
        # filter topk connections
        sim_items_slices = []
        sim_weights_slices = []
        i = 0
        while i < similarity.shape[0]:
            similarity = similarity[i:, :]
            sim = similarity[:256, :]
            sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
            sim_weights = np.take_along_axis(sim, sim_items, axis=1)
            sim_items_slices.append(sim_items)
            sim_weights_slices.append(sim_weights)
            i = i + 256
        sim = similarity[256:, :]
        sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
        sim_weights = np.take_along_axis(sim, sim_items, axis=1)
        sim_items_slices.append(sim_items)
        sim_weights_slices.append(sim_weights)

        sim_items = np.concatenate(sim_items_slices, axis=0)
        sim_weights = np.concatenate(sim_weights_slices, axis=0)
        row = []
        col = []
        for i in range(len(sim_items)):
            row.extend([i]*len(sim_items[i]))
            col.extend(sim_items[i])
        values = sim_weights / (sim_weights.sum(axis=1, keepdims=True)+1e-24)
        values = np.nan_to_num(values).flatten()
        adj_mat = csr_matrix((values, (row, col)), shape=(
            dataset.item_num + 1, dataset.item_num + 1))
        g = dgl.from_scipy(adj_mat, 'w')
        g.edata['w'] = g.edata['w'].float()
        # print("saving isim graph to binary file...")
        # dgl.save_graphs(graph_file, [g])
        return g


class NEWV2Dataset(Dataset):
    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        self.pretrained_model = config["pretrained_model"]
        self._load_pretrained_model(self.pretrained_model)
        dataset_name = config['dataset']
        if dataset_name == 'ml-100k':
            self.theta = 0.6
        elif dataset_name == 'beauty':
            self.theta = 0.3
        elif dataset_name == 'sports':
            self.theta = 0.9
        elif dataset_name == 'retailrocket-view':
            self.theta = 0.6
        else:
            self.theta = 0.6

        super().__init__(config)
        if config["benchmark_filename"] is not None:
            self._benchmark_presets()
    
    def _load_pretrained_model(self, model_file):
        checkpoint = torch.load(model_file)
        config = checkpoint["config"]
        dataset = create_dataset(config)
        train_data, _, _ = data_preparation(config, dataset)
        model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
        self.pretrained_model = model

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        super()._change_feat_format()

        if self.config["benchmark_filename"] is not None:
            return
        self.logger.debug("Augmentation for sequential recommendation.")
        self.data_augmentation()

    def _aug_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(
                    list_field, list_ftype, FeatureSource.INTERACTION, list_len
                )

        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )

    def data_augmentation(self):
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                new_dict[list_field] = torch.zeros(
                    shape, dtype=self.inter_feat[field].dtype
                )

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

    def _benchmark_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )
        self.inter_feat[self.item_list_length_field] = self.inter_feat[
            self.item_id_list_field
        ].agg(len)

    def inter_matrix(self, form="coo", value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                "dataset does not exist uid/iid, thus can not converted to sparse matrix."
            )

        l1_idx = self.inter_feat[self.item_list_length_field] == 1
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config["LIST_SUFFIX"]
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]]
                )
            elif (not field.endswith(list_suffix)) and (
                field != self.item_list_length_field
            ):
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field]]
                )
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(
            local_inter_feat, self.uid_field, self.iid_field, form, value_field
        )

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of built :class:`Dataset`.
        """
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )
        train, val, test = super().build()
        inter_feat = train.inter_feat
        inter_feat = inter_feat.to(self.pretrained_model.device)
        new_dict = {
            'interest_entropy': None,
        }
        # import pdb
        # pdb.set_trace()
        interest_entropy = self.interest_entropy(inter_feat[train.iid_field + self.config['LIST_SUFFIX']], inter_feat[train.item_list_length_field])
        new_dict['interest_entropy'] = interest_entropy
        inter_feat.update(Interaction(new_dict))
        train.inter_feat = inter_feat
        return train, val, test

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
        embeddings = self.pretrained_model.item_embedding(item_seq)
        batch_size, num_items, _ = embeddings.shape
        
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.bmm(embeddings_norm, embeddings_norm.transpose(1, 2))
        adj_matrix = (sim_matrix > theta).float()

        rows = torch.arange(num_items, device=self.pretrained_model.device).view(1, num_items, 1)  #
        cols = torch.arange(num_items, device=self.pretrained_model.device).view(1, 1, num_items)  #
        mask = (rows >= item_seq_len.view(batch_size, 1, 1)) & (cols >= item_seq_len.view(batch_size, 1, 1))  #
        adj_matrix[mask] = 0
        cluster_num = self.get_connected_components_sizes(adj_matrix, item_seq_len)
        
        batch_class_probs = []
        for b, l in zip(range(batch_size), item_seq_len):
            c_n = cluster_num[b]
            c_p = torch.FloatTensor([_ / l for _ in c_n])
            batch_class_probs.append(c_p)
        ie_values = []
        for class_probs in batch_class_probs:
            entropy = -torch.sum(class_probs * torch.log2(class_probs + 1e-10))
            ie_values.append(entropy)
            # ie_values.append(torch.tensor(len(class_probs)))
        
        return torch.stack(ie_values)