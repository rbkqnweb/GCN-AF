import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from FeatUtils import setAxeFontSize
from utils.preprocess_help import load_data

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_counts(features, labels, target_label):
    target_features = features[labels == target_label]
    sample_counts = np.sum(target_features != 0, axis=0)
    sample_counts_pct = sample_counts*1.0 / target_features.shape[0]

    plt.figure(figsize=(12, 6))

    # 绘制条形图
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(sample_counts) + 1), sample_counts)
    # plt.bar(range(1, len(sample_counts) + 1), sample_counts_pct)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Sample Count')
    plt.grid(axis='y', alpha=0.75)
    plt.title(f'Sample Count in Feature Dimension for Label {target_label}')

    # 绘制频率分布直方图
    plt.subplot(1, 2, 2)
    plt.hist(sample_counts, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Sample Count')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution of Sample Counts')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
dataset_str = 'cora'
adj, features, labels, _, _, _ = load_data(dataset_str)

num_feat = np.array(features.todense())
num_label = np.array(labels.argmax(axis=1))

# 选择要统计的类别
# 818
target_label = 3  # 选择类别3进行绘制
target_features = num_feat[num_label == target_label]
zero_sample_counts = np.sum(target_features == 0, axis=0)
zero_count = np.sum(zero_sample_counts >= 818*0.9)
print()
# 调用函数进行统计和绘图
# plot_feature_counts(num_feat, num_label, target_label)
target_features = num_feat[num_label == target_label]
sample_counts = np.sum(target_features != 0, axis=0)
sample_counts_pct = sample_counts*1.0 / target_features.shape[0]
# 设置阈值
threshold = 0.1

# 统计大于阈值的元素个数
count = np.sum(sample_counts_pct > threshold)
print("count=\n",count)
print("count ratio=\n",count/len(sample_counts_pct))
# 将 y 轴标签格式化为百分数形式
def percent_formatter(x, pos):
    return '{:.0%}'.format(x)



# plt.figure(figsize=(12, 6))
fig, ax = plt.subplots(figsize=(12, 6))
# plt.bar(range(1, len(sample_counts) + 1), sample_counts,width=1)
# ax.bar(range(1, len(sample_counts) + 1), sample_counts_pct,width=5)
ax.bar(range(1, len(sample_counts) + 1),sample_counts_pct,alpha=0.7,width=5)
# ax.bar(range(1, len(sample_counts) + 1), 1-sample_counts_pct,color='r',alpha=0.7,width=5,bottom=sample_counts_pct)
ax.set_xlabel('Attribute')
ax.set_ylabel('Sample Ratio(%)')
ax.grid(axis='y', alpha=0.75)
ax.set_title(f'Sample Ratio in Each Attribute for Label {target_label}')
ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
setAxeFontSize(ax,title_size=12,axis_size=12,tick_size=12)

plt.show()