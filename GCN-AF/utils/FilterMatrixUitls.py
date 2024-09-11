"""
    Get the probability matrix based on ParkWalks
"""
import itertools
import os
import random

import numpy as np
from numpy.linalg import inv
from scipy.sparse import csr_matrix, dok_matrix
from scipy import sparse

from utils.preprocess_help import load_data


def getParWalksProbMatrix(adj, alpha=1e-6):
    A_tilde = adj.toarray() + np.identity(adj.shape[0])
    D = A_tilde.sum(axis=1)
    A_ = np.diag(D ** -0.5).dot(A_tilde).dot(np.diag(D ** -0.5))

    Lambda = np.identity(len(A_))
    L = np.diag(D) - adj
    P = inv(L + alpha * Lambda)
    return P


###################PPMI#########################

def _generate_path(node_id, dict_nid_neighbors, re, path_len):
    path_node_list = [node_id]
    for i in range(path_len - 1):
        temp = dict_nid_neighbors.get(path_node_list[-1])
        if len(temp) < 1:
            break
        else:
            path_node_list.append(random.choice(temp))
    # update difussion matrix re
    # 来自 itertools 模块的函数 combinations(list_name, x) 将一个列表和数字 x 作为参数，
    # 并返回一个元组列表，每个元组的长度为 x ，其中包含x个元素的所有可能组合。
    # 列表中元素不能与自己结合，不包含列表中重复元素
    for pair in itertools.combinations(path_node_list, 2):
        if pair[0] == pair[1]:
            re[pair[0], pair[1]] += 1.0
        else:
            re[pair[0], pair[1]] += 1.0
            re[pair[1], pair[0]] += 1.0


def _diffusion_fun_sampling(A, sampling_num=100, path_len=3, self_loop=True, spars=False):
    # the will return diffusion matrix
    re = None
    if not spars:
        re = np.zeros(A.shape)
    else:
        re = sparse.dok_matrix(A.shape, dtype=np.float32)

    if self_loop:
        A_with_selfloop = A + sparse.identity(A.shape[0], format="csr")
    else:
        A_with_selfloop = A

    # 生成存储每个节点的一阶邻居集合的Dictionary
    # record each node's neignbors
    dict_nid_neighbors = {}
    for nid in range(A.shape[0]):
        neighbors = np.nonzero(A_with_selfloop[nid])[1]
        # use cuda
        # neighbors = torch.nonzero(A_with_selfloop[nid],as_tuple=True)
        dict_nid_neighbors[nid] = neighbors

    # for each node
    for i in range(A.shape[0]):
        # for each sampling iter
        for j in range(sampling_num):  # 以节点i为起点，进行sampling_num次随机游走
            _generate_path(i, dict_nid_neighbors, re, path_len)  # 一次以i为起始点的随机游走
    return re


def calc_pmi(counts, cds=0.75):
    """
    Calculates e^PMI; PMI without the log().
    counts --> a sparse matrix (CSR) 对应于频率矩阵
    计算的只是 p(w,c) 与 p(w)*p_alpha(c)的比值
    """
    sum_w = np.array(counts.sum(axis=1))[:, 0]
    sum_c = np.array(counts.sum(axis=0))[0, :]
    if cds != 1:
        sum_c = sum_c ** cds
    sum_total = sum_c.sum()
    # np.reciprocal(x) --> Calculates ``1/x``.
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)

    pmi = csr_matrix(counts)
    pmi = multiply_by_rows(pmi, sum_w)
    pmi = multiply_by_columns(pmi, sum_c)
    pmi = pmi * sum_total
    return pmi


def multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())
