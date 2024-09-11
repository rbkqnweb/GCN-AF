import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 创建示例数据，假设features是一个2708x1433的特征矩阵
from scipy.interpolate import make_interp_spline

from plotdemo import linestylelist
from utils.preprocess_help import load_data

'''
    方差分析
'''


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# dataset_str = 'cora'
# # dataset_str = 'citeseer'
# adj, features, labels, _, _, _ = load_data(dataset_str)
#
# num_feat = np.array(features.todense())
# num_label = np.array(labels.argmax(axis=1))
# num_features = num_feat.shape[1]
#
# # 将特征矩阵转换为DataFrame
# df = pd.DataFrame(num_feat, columns=[f'feature_{i}' for i in range(num_features)])
#
# # 计算Pearson相关系数矩阵
# corr_matrix = df.corr()
#
# # 绘制相关性矩阵热力图
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
# plt.title('Pearson Correlation Heatmap of Features')
# plt.show()


# # 创建示例数据，假设features是一个2708x1433的特征矩阵
# num_samples = 2708
# num_features = 1433
# features = np.random.randint(0, 2, size=(num_samples, num_features))

# target_label = 3  # 选择类别3进行绘制
# target_features = num_feat[num_label == target_label]
# num_features = target_features.shape[1]
def calcVar(feat_mat):
    df = pd.DataFrame(feat_mat, columns=[f'feature_{i}' for i in range(feat_mat.shape[1])])
    return df.var()


# 将特征矩阵转换为DataFrame
# df = pd.DataFrame(num_feat, columns=[f'feature_{i}' for i in range(num_features)])
# # df = pd.DataFrame(target_features, columns=[f'feature_{i}' for i in range(num_features)])
#
# # 计算特征的方差
# feature_variances = df.var()
# num_zero_variance_features = (feature_variances == 0).sum()
#
# print("方差为0的特征数：", num_zero_variance_features)
# 绘制特征方差分布图
# plt.figure(figsize=(10, 6))
# plt.hist(feature_variances, bins=50, color='skyblue', edgecolor='black')
# plt.xlabel('Feature Variance')
# plt.ylabel('Frequency')
# plt.title('Distribution of Feature Variances')
# plt.grid(axis='y', alpha=0.75)
# plt.show()

def setAxeFontSize(ax):
    tick_label_fontsize = 12
    axis_label_fontsize = 12
    title_fontsize = 14

    ax.set_title(label=ax.get_title(), fontsize=title_fontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=axis_label_fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=axis_label_fontsize)
    ax.tick_params(labelsize=tick_label_fontsize)


def PlotVarOnTwoGraph(feat_var_mat):
    # 创建带有两个子图的特征方差分布图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # 第一个子图：完整特征方差分布图
    bin_counts, bin_edges = np.histogram(feat_var_mat[0], bins='sturges')
    # 为了绘制趋势曲线，我们需要计算每个bin的中心点作为横坐标
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 使用 scipy 的 make_interp_spline 创建一个平滑的曲线
    spl = make_interp_spline(bin_centers, bin_counts, k=3)  # k 是平滑曲线的度
    # 创建更细的横坐标点，用于绘制平滑曲线
    fine_bin_centers = np.linspace(bin_centers.min(), bin_centers.max(), 1400)
    # 计算细分横坐标点对应的曲线值
    smooth_bin_counts = spl(fine_bin_centers)

    # 绘制平滑的趋势曲线
    axs[0].plot(fine_bin_centers, smooth_bin_counts, color="red", linestyle=linestylelist[1], linewidth=2, label="Trend Line")

    axs[0].hist(feat_var_mat[0], bins='sturges', color='skyblue', edgecolor='black')
    axs[0].set_xlabel('Attribute Variance')
    axs[0].set_ylabel('Number of Attributes')
    # axs[0].set_title('Distribution of Label 3 Feature Variances (Cora)')
    axs[0].set_title('Distribution of Attribute Variance (Cora)')
    axs[0].grid(axis='y', alpha=0.75)
    # 第二个子图：放大特征方差小于1的部分

    bin_counts, bin_edges = np.histogram(feat_var_mat[1], bins='sturges')
    # 为了绘制趋势曲线，我们需要计算每个bin的中心点作为横坐标
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 使用 scipy 的 make_interp_spline 创建一个平滑的曲线
    spl = make_interp_spline(bin_centers, bin_counts, k=3)  # k 是平滑曲线的度
    # 创建更细的横坐标点，用于绘制平滑曲线
    fine_bin_centers = np.linspace(bin_centers.min(), bin_centers.max(), 1400)
    # 计算细分横坐标点对应的曲线值
    smooth_bin_counts = spl(fine_bin_centers)

    # 绘制平滑的趋势曲线
    axs[1].plot(fine_bin_centers, smooth_bin_counts, color="red", linestyle=linestylelist[1], linewidth=2, label="Trend Line")


    axs[1].hist(feat_var_mat[1], bins='sturges', color='skyblue', edgecolor='black')
    axs[1].set_xlabel('Attribute Variance')
    axs[1].set_ylabel('Number of Attributes')
    # axs[1].set_title('Distribution of Label 3 Feature Variances (CiteSeer)')
    axs[1].set_title('Distribution of Attribute Variance (CiteSeer)')
    axs[1].grid(axis='y', alpha=0.75)
    setAxeFontSize(axs[0])
    setAxeFontSize(axs[1])
    plt.tight_layout()
    plt.show()


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
dataset_str = 'cora'
# num_label = np.array(labels.argmax(axis=1))
_, cora_features, cora_labels, _, _, _ = load_data(dataset_str)
dataset_str = 'citeseer'
_, citeseer_features, citeseer_labels, _, _, _ = load_data(dataset_str)

target_label = 3  # 选择类别3进行绘制
num_label_cora = np.array(cora_labels.argmax(axis=1))
num_label_citeseer = np.array(citeseer_labels.argmax(axis=1))

num_feat_cora = np.array(cora_features.todense())
target_features_cora = num_feat_cora[num_label_cora == target_label]
num_feat_citeseer = np.array(citeseer_features.todense())
target_features_citeseer = num_feat_citeseer[num_label_citeseer == target_label]
# feat_var_cora = calcVar(target_features_cora)
# feat_var_citeseer = calcVar(target_features_citeseer)
feat_var_cora = calcVar(num_feat_cora)
feat_var_citeseer = calcVar(num_feat_citeseer)
# print('Cora Samples of Label 3:',target_features_cora.shape)
# print('CiteSeer Samples of Label 3:',target_features_citeseer.shape)
# _, bins_cora =np.histogram(feat_var_cora,bins='sturges')
# _, bins_citeseer =np.histogram(feat_var_citeseer,bins='sturges')
# print('bins_cora=\n',bins_cora)
# print('bins_citeseer=\n',bins_citeseer)

PlotVarOnTwoGraph([feat_var_cora, feat_var_citeseer])
