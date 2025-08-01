import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import scipy.sparse as sp
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import random
from torch_geometric.data import Data

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def lossFunction(recon_x, x, mu, log_var):
    # bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    bce = F.mse_loss(recon_x, x)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj, device):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized).to(device)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def get_norm(adj_train):
    adj = adj_train
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    # pos_weight =  torch.sparse.FloatTensor(torch.FloatTensor(pos_weight))
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    return adj_label, torch.tensor(pos_weight), norm

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.tensor(features)

def adjust_learning_rate(optimizer, epoch, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def get_pos_weight(adj, device):
#     pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#     return pos_weight.to(device)

def for_gae(features, adj, device):
    n_nodes, feat_dim = features.shape
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj, device)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    return adj_label.to(device), torch.tensor(norm).to(device), torch.tensor(pos_weight).to(device)

def compute_accuracy_teacher_mask(prediction, label, mask):
    correct = 0
    indices = torch.nonzero(mask)
    for i in indices:
        if prediction[i] == label[i]:
            correct += 1
    accuracy = correct / len(prediction) * 100
    return accuracy

def compute_accuracy_teacher(prediction, label):
    correct = 0
    # label = torch.argmax(label, dim=1)
    for i in range(len(label)):
        if prediction[i] == label[i]:
            correct += 1
    accuracy = correct / len(prediction) * 100
    return accuracy


import numpy as np
import torch
import networkx as nx
from sklearn.metrics import pairwise_distances
import community as community_louvain


def sample_subgraph(adj_matrix, features, sampling_type="community", target_size=200):
    """
    从密集邻接矩阵和特征矩阵采样子图

    参数:
    adj_matrix: 密集邻接矩阵 (numpy或torch.Tensor) [N, N]
    features: 节点特征矩阵 [N, F]
    sampling_type: 采样类型 ("community", "pagerank", "random_walk")
    target_size: 目标子图节点数

    返回:
    sub_adj: 子图邻接矩阵 [target_size, target_size]
    sub_features: 子图特征矩阵 [target_size, F]
    """
    # 转换为NetworkX图
    G = nx.from_numpy_array(adj_matrix.cpu().numpy() if torch.is_tensor(adj_matrix) else adj_matrix)

    if sampling_type == "community":
        return community_based_sampling(G, features, target_size)
    elif sampling_type == "pagerank":
        return pagerank_sampling(G, features, target_size)
    elif sampling_type == "random_walk":
        return random_walk_sampling(G, features, target_size)
    else:
        raise ValueError(f"未知采样类型: {sampling_type}")


def community_based_sampling(G, features, target_size):
    """社区结构感知采样"""
    # 使用Louvain算法检测社区
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # 按社区大小比例采样
    sampled_nodes = []
    total_nodes = len(G)

    for comm_id, nodes in communities.items():
        ratio = len(nodes) / total_nodes
        n_sample = max(1, int(target_size * ratio))

        # 在社区内均匀采样
        comm_sample = np.random.choice(nodes, size=n_sample, replace=False)
        sampled_nodes.extend(comm_sample)

    # 确保采样节点数符合目标
    if len(sampled_nodes) > target_size:
        sampled_nodes = np.random.choice(sampled_nodes, size=target_size, replace=False)
    elif len(sampled_nodes) < target_size:
        # 补充随机节点
        extra = np.random.choice(list(G.nodes), size=target_size - len(sampled_nodes), replace=False)
        sampled_nodes = np.concatenate([sampled_nodes, extra])

    return extract_subgraph(sampled_nodes, G, features)


def pagerank_sampling(G, features, target_size):
    """PageRank重要性采样"""
    pr = nx.pagerank(G)
    sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)

    # 分层采样
    high = [n for n, _ in sorted_nodes[:int(target_size * 0.5)]]
    mid = [n for n, _ in sorted_nodes[int(target_size * 0.5):int(target_size * 0.8)]]
    low = [n for n, _ in sorted_nodes[-int(target_size * 0.2):]]

    sampled_nodes = np.array(high + mid + low)
    return extract_subgraph(sampled_nodes, G, features)


def random_walk_sampling(G, features, target_size, walk_length=40, restart_prob=0.2):
    """随机游走采样"""
    sampled_nodes = set()
    current = np.random.choice(list(G.nodes))

    while len(sampled_nodes) < target_size:
        if np.random.rand() < restart_prob or not list(G.neighbors(current)):
            current = np.random.choice(list(G.nodes))
        else:
            current = np.random.choice(list(G.neighbors(current)))
        sampled_nodes.add(current)

    return extract_subgraph(list(sampled_nodes), G, features)


def extract_subgraph(sampled_nodes, G, features):
    """
    从原始图中提取子图
    """
    # 对节点索引排序
    sampled_nodes = sorted(sampled_nodes)

    # 创建子图
    subgraph = G.subgraph(sampled_nodes)

    # 获取邻接矩阵
    adj_matrix = nx.to_numpy_array(subgraph)

    # 获取特征矩阵
    if torch.is_tensor(features):
        sub_features = features[sampled_nodes]
    else:
        sub_features = features[sampled_nodes, :]

    return sp.coo_matrix(torch.tensor(adj_matrix)), sub_features


import torch.nn as nn
from torch import Tensor


class SemanticConsistency(nn.Module):
    """
    Semantic consistency loss is introduced by
    `CyCADA: Cycle-Consistent Adversarial Domain Adaptation (ICML 2018) <https://arxiv.org/abs/1711.03213>`_

    This helps to prevent label flipping during image translation.

    Args:
        ignore_index (tuple, optional): Specifies target values that are ignored
            and do not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: ().
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> loss = SemanticConsistency()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, ignore_index=(), reduction='mean'):
        super(SemanticConsistency, self).__init__()
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        for class_idx in self.ignore_index:
            target[target == class_idx] = -1
        return self.loss(input, target)


def entropy_loss_f(logit):
    probs_t = F.softmax(logit, dim=-1)
    probs_t = torch.clamp(probs_t, min=1e-9, max=1.0)
    entropy_loss = torch.mean(torch.sum(-probs_t * torch.log(probs_t), dim=-1))
    return entropy_loss

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss






