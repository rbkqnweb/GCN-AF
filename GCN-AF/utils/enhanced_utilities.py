import os

import numpy as np
import scipy.sparse as sp

from utils.FilterMatrixUitls import _diffusion_fun_sampling, calc_pmi
from utils.preprocess_help import load_data


# Calculate the corresponding percentile according to the passed-in parameters


def getCentileValue(probs, upper_percent=90):
    a = []
    for i in range(probs.shape[0]):
        for j in range(i + 1, probs.shape[0]):
            if probs[i, j] != 0:
                a.append(probs[i, j])
    upper_bound_value = np.percentile(a, upper_percent)
    return upper_bound_value


# get parwalks filter matrix
def getFilterMatrix(probs, upper_percent=90, beta=0.1):
    upper_bound_value = getCentileValue(probs, upper_percent)
    filter_matrix = sp.csr_matrix(np.identity((probs.shape[0])))
    for i in range(probs.shape[0]):
        for j in range(i + 1, probs.shape[0]):
            if probs[i, j] >= upper_bound_value:
                filter_matrix[i, i] = 1 - beta
                filter_matrix[i, j] = beta
                filter_matrix[j, j] = 1 - beta
                filter_matrix[j, i] = beta
    return filter_matrix


#############################PPMI###########################################
'''
    对于ppmi的修改 using context distribution smoothing to improve performance
    解决的问题：防止出现采样次数比较少 反而最后的ppmi的元素值比较大的情况
'''


# get ppmi matrix
def getPPMIMatrix(adj, sampling_num=100, path_len=2,
                        self_loop=True, spars=True,cds=0.75):
    '''

    :param adj: csr_matrix
    :param sampling_num:
    :param path_len:
    :param self_loop:
    :param spars:
    :param cds:
    :return:
    '''
    # print("Do the sampling...")
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # adj, _, _, _, _, _ = load_data(dataset_str)
    mat = _diffusion_fun_sampling(
        adj, sampling_num=sampling_num, path_len=path_len,
        self_loop=self_loop, spars=spars)
    print("Calculating the PPMI...")
    pmi = calc_pmi(counts=mat, cds=cds)
    return pmi


