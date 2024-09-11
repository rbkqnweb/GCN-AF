import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import KernelPCA, SparsePCA
from torch import nn, optim
from torch.nn import init

from utils.preprocess_help import preprocess_adj, get_sparse_input


def sparse_kernel_pca(sparse_data, n_components=2):
    # 将tensor稀疏数据转换为稀疏矩阵格式
    sparse_data_coo = sparse_data.coalesce()
    rows = sparse_data_coo.indices()[0]
    cols = sparse_data_coo.indices()[1]
    values = sparse_data_coo.values()
    sparse_data_csr = csr_matrix((values, (rows, cols)), shape=sparse_data.shape)

    # 将稀疏矩阵转换为密集的NumPy数组
    dense_data = sparse_data_csr.toarray()

    # 初始化核PCA模型
    kernel_pca = KernelPCA(kernel="linear", n_components=n_components)

    # 拟合核PCA模型并进行降维
    transformed_data = kernel_pca.fit_transform(dense_data)

    # 将降维后的数据转换为稀疏矩阵格式
    sparse_transformed_data = csr_matrix(transformed_data)

    # 将稀疏矩阵转换为稀疏tensor
    rows, cols = sparse_transformed_data.nonzero()
    values = sparse_transformed_data.data
    sparse_transformed_data_tensor = torch.sparse_coo_tensor(torch.LongTensor([rows, cols]),
                                                             torch.FloatTensor(values),
                                                             size=sparse_transformed_data.shape)

    return sparse_transformed_data_tensor


# dropout
def sparse_dropout(x, rate, noise_shape, training=False):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    if not training:
        return x
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()
    i = x._indices()  # [2, 49216]
    v = x._values()  # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1. / (1 - rate))

    return out


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer, n_feat_nonzero, dropout=0, device=torch.device('cuda:0')):
        super(SimpleGCN, self).__init__()

        self.device = device
        self.dropout = dropout

        self.n_feat_nonzero = n_feat_nonzero
        self.n_layer = n_layer
        self.lin = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.lin)

    def forward(self, features, adj, enhanced_message_matrix=None):
        adj_drop = adj
        supports = preprocess_adj(adj_drop)
        X, A = get_sparse_input(features, supports, self.device)
        X = sparse_dropout(X, self.dropout, self.n_feat_nonzero, self.training)
        X = torch.sparse.mm(X, self.lin)
        layerwise_feat_list = []
        for i in range(self.n_layer):
            X = torch.sparse.mm(A, X)
            # filter matrix plays a role in middle layers
            if enhanced_message_matrix is not None and i != 0 and i != self.n_layer - 1:
                X = torch.sparse.mm(enhanced_message_matrix, X)
            layerwise_feat_list.append(X)
        return F.log_softmax(X, dim=1), X, layerwise_feat_list


# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def pretrainingAE(feat_matrix, num_feats, hidden_dim):
    # 定义模型、损失函数和优化器
    model = Autoencoder(input_dim=num_feats, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 训练模型
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        encoded, decoded = model(feat_matrix)
        loss = criterion(decoded, feat_matrix)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # 提取编码器的输出
    model.eval()
    with torch.no_grad():
        encoded, _ = model(feat_matrix)
        print(f'Encoded features shape: {encoded.shape}')
    return encoded


def get_sparse_AEEncoderOutput(sparse_data, n_components=2):
    # 将tensor稀疏数据转换为稀疏矩阵格式
    sparse_data_coo = sparse_data.coalesce()
    rows = sparse_data_coo.indices()[0]
    cols = sparse_data_coo.indices()[1]
    values = sparse_data_coo.values()
    sparse_data_csr = csr_matrix((values, (rows, cols)), shape=sparse_data.shape)

    # 将稀疏矩阵转换为密集的NumPy数组
    dense_data = sparse_data_csr.toarray()

    # 进行预训练
    encoded = pretrainingAE(dense_data, dense_data.shape[1], hidden_dim=n_components)

    # 将降维后的数据转换为稀疏矩阵格式
    sparse_transformed_data = csr_matrix(encoded)

    # 将稀疏矩阵转换为稀疏tensor
    rows, cols = sparse_transformed_data.nonzero()
    values = sparse_transformed_data.data
    sparse_transformed_data_tensor = torch.sparse_coo_tensor(torch.LongTensor([rows, cols]),
                                                             torch.FloatTensor(values),
                                                             size=sparse_transformed_data.shape)

    return sparse_transformed_data_tensor


class SimpleGCNAF(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer, n_feat_nonzero, dropout=0, device=torch.device('cuda:0'), usePCA=False,
                 useAE=False):
        super(SimpleGCNAF, self).__init__()

        self.device = device
        self.dropout = dropout

        self.n_feat_nonzero = n_feat_nonzero
        self.n_layer = n_layer
        self.lin = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.usePca = usePCA
        self.useAE = useAE
        self.filtered_dim = in_dim
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.lin)

    def forward(self, features, adj, enhanced_message_matrix=None):
        adj_drop = adj
        supports = preprocess_adj(adj_drop)
        X, A = get_sparse_input(features, supports, self.device)
        if self.training:
            X = sparse_dropout(X, self.dropout, self.n_feat_nonzero, self.training)
        if self.pcaFirst == True:
            X = sparse_kernel_pca(X, n_components=self.filtered_dim)
        layerwise_feat_list = []
        for i in range(self.n_layer):
            X = torch.sparse.mm(A, X)
            if self.usePca == True:
                # 使用pca方法进行属性滤波
                X = sparse_kernel_pca(X, n_components=self.filtered_dim)
            if self.useAE == True:
                X = get_sparse_AEEncoderOutput(X, n_components=self.filtered_dim)
            # filter matrix plays a role in middle layers
            if enhanced_message_matrix is not None and i != 0 and i != self.n_layer - 1:
                X = torch.sparse.mm(enhanced_message_matrix, X)
            layerwise_feat_list.append(X)
        X = torch.sparse.mm(X, self.lin)
        return F.log_softmax(X, dim=1), X, layerwise_feat_list


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_feat_nonzero, dropout=0, is_sparse_input=False, activision=F.relu, bias=True):
        super(GCNLayer, self).__init__()

        self.dropout = dropout
        self.activision = activision
        self.n_feat_nonzero = n_feat_nonzero
        self.is_sparse_input = is_sparse_input
        # self.lin = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.lin = nn.Parameter(torch.randn(in_dim, out_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # 自定义参数初始化方式
        # 权重参数初始化方式
        init.kaiming_uniform_(self.lin)
        if self.bias:  # 偏置参数初始化为0
            init.zeros_(self.bias)

    # full-batch X&A
    def forward(self, inputs):
        X, A = inputs
        if self.is_sparse_input:
            X = sparse_dropout(X, self.dropout, self.n_feat_nonzero, self.training)
        else:
            X = F.dropout(X, self.dropout, self.training)
        if self.is_sparse_input:
            X = torch.sparse.mm(X, self.lin)
        else:
            X = torch.mm(X, self.lin)
        out = torch.sparse.mm(A, X)
        if self.bias is not None:
            out += self.bias
        return self.activision(out), A


class GCN_Model(nn.Module):
    def __init__(self, dim_list, n_feat_nonzero, dropout=0, activision=F.relu, device=torch.device('cpu')):
        super(GCN_Model, self).__init__()

        self.device = device
        self.dropout = dropout
        dims = [(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)]
        layers = [GCNLayer(d[0], d[1], n_feat_nonzero, dropout, False, activision, False) for d in dims]
        layers[0].is_sparse_input = True
        self.layers = nn.ModuleList(layers)

    def forward(self, features, adj, enhanced_message_matrix=None):
        # features: normalized features
        # adj: raw adj matrix (in spicy.csr_matrix format)

        supports = preprocess_adj(adj)
        X, A = get_sparse_input(features, supports, self.device)
        # for l in self.layers:
        #     X, A_ = l((X, A))
        # return F.log_softmax(X, dim=1), X
        layerwise_feat_list = []
        for i in range(len(self.layers)):
            if enhanced_message_matrix is not None and i != 0 and i != len(self.layers) - 1:
                X = torch.sparse.mm(enhanced_message_matrix, X)
            X, _ = self.layers[i]((X, A))
            layerwise_feat_list.append(X)
        return F.log_softmax(X, dim=1), X, layerwise_feat_list
