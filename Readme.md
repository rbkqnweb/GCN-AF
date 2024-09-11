### 原始数据及说明
1. data文件夹存储了原始数据
2. 其中cora_citeseer文件夹包含了Cora与CiteSeer数据集、new_data文件夹包含Chameleon、Cornell、Actor、Squirrel、Texas与Wisconsin数据集
3. new_data文件夹中，各个数据集文件夹中的out1_graph_edges.txt文件表示图的边信息，out1_node_feature_label.txt文件表示图节点的特征与标签信息

### 从原始数据生成新数据说明

1. 主要所做的操作是将图数据从文件中加载进来，获取图的邻接矩阵、节点的特征向量以及节点的标签的信息

2. 生成属性滤波器的方法为PCA方法或者AE预训练方法，两种方法较为简单，所以代码直接获取了原始特征矩阵经过PCA滤波器或者AE滤波器处理后的结果，并没有将滤波器持久化为一个文件

### GCN-AF代码说明及运行环境
1. GCN-AF文件夹包含了GCN-AF模型需要的运行程序及其他工具文件，具体情况如下：

   - `data/`、`new_data/`文件夹包括了Cora、CiteSeer在内的多个数据集文件
   - `splits/`文件夹包括各个数据集的10次不同的训练集、验证集和测试集的划分
   - `utils/`文件夹包括数据处理以及模型定义等多个文件、其中`models.py`文件中具体定义了**GCN-AF(PCA)**与**GCN-AF(AE)**模型
   - `FeatAnalysis.py`、`FeatCountAnalysis.py`主要分析了原始的特征矩阵中无效属性的问题
   - `FeatParamAnalysis.py`对GCN-AF模型的维度参数进行了分析
   - `utils_xx.py`主要包括数据预处理的相关代码
   - `PlotFeatAcc.py`主要绘制了4种综合模型GCN-SF-AF的在多个数据集上的准确率柱状图
   - `GCN-AFTraining.py`为主程序文件，主要定义了GCN-AF模型的具体训练代码，键入`python GCN-AFTraining.py `即可进行GCN-AF模型的训练
   - `PlotClusterScore.py`绘制了两种GCN-AF模型的输出在两种聚类指标上的结果

2. 运行环境说明

    程序主要依托于`Python 3.6.8`运行，主要的依赖环境如下：

  * numpy==1.19.4
  * scipy==1.5.4
  * matplotlib==3.3.4
  * networkx==2.5
  * torch==1.8.0+cu111
  * scikit_learn==1.3.0

  另外，本项目所运用的`CUDA`版本为`11.1`，更多的依赖环境信息，详见`requirements.txt`文件

### 实验结果

`result`文件中记录了主要的实验结果，具体包括：

- `feat_param.png`描述了GCN-AF模型的参数分析结果
- `feat_Cora.png`、`feat_CiteSeer.png`文件分别刻画了GCN-AF模型在Cora、CiteSeer数据集上的聚类效果

### 对比模型可参考
https://github.com/tkipf/gcn
https://github.com/PetarV-/GAT