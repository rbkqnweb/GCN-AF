from utils.FilterMatrixUitls import getParWalksProbMatrix
from utils.enhanced_utilities import getFilterMatrix, getPPMIMatrix
from utils.getAdj import getAdjMatrix


def generatingParWalksFilterMatrix(dataset_str, beta=0.1, upper_percent=99.9):
    """
    生成ParWalks消息强化矩阵
    :param dataset_str: 数据集名称
    :param beta: 增强强度
    :param upper_percent: 阈值
    :return: ParWalks消息强化矩阵
    """
    splits_file_path = 'splits/' + dataset_str + '_split_0.6_0.2_0.npz'
    adj = getAdjMatrix(dataset_str=dataset_str, splits_file_path=splits_file_path)
    P = getParWalksProbMatrix(adj=adj)
    filter_matrix = getFilterMatrix(P, upper_percent=upper_percent, beta=beta)
    # print(dataset_str + "_parwalks_filter_matrix.nnz:\n", filter_matrix.nnz)
    return filter_matrix


def generatingPPMIFilterMatrix(dataset_str, beta=0.1, upper_percent=90.0, sampling_num=100, path_len=7, cds=0.75):
    '''
    生成PPMI消息强化矩阵
    :param dataset_str: 数据集名称
    :param sampling_num: 采样数
    :param path_len: 路径长度
    :param self_loop: 自环
    :param spars: 矩阵是否稀疏
    :param cds: cds参数
    :return: PPMI消息强化矩阵
    '''
    splits_file_path = 'splits/' + dataset_str + '_split_0.6_0.2_0.npz'
    adj = getAdjMatrix(dataset_str, splits_file_path=splits_file_path)
    P = getPPMIMatrix(adj, sampling_num=sampling_num, path_len=path_len, cds=cds)
    filter_matrix = getFilterMatrix(P, upper_percent=upper_percent, beta=beta)
    # print("ppmi_filter_matrix.nnz:\n", filter_matrix.nnz)
    return filter_matrix
