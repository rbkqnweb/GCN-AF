import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.optim import Adam

import utils_data
from utils.DataStorage import load_sparse_matrix
from utils.TrainingUtils import train
from utils.models import SimpleGCN, GCN_Model, SimpleGCNAF
from utils_old_data import sparse_to_tuple


def getBatchTrainValTestSet(dataset_str,
                            batch=0):
    """
    get different TrainValTestSet
    :param dataset_str:
    :param batch: batch number of the spliting
    :return: train_mask,val_mask,test_mask
    """
    splits_file_path = 'splits/' + str(dataset_str) + '_split_0.6_0.2_' + str(batch) + '.npz'
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    return train_mask.astype(bool), val_mask.astype(bool), test_mask.astype(bool)


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    convert scipy.sparse matrix into torch.sparse Tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


dataset_strs = ['cora', 'citeseer', 'chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin']
dataset = 'cora'
device = torch.device('cpu')

batch = 2
dataset_split = 'splits/' + str(dataset) + '_split_0.6_0.2_' + str(batch) + '.npz'
g, adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = utils_data.load_data(
    dataset, dataset_split)
# print(features)
features = sparse_to_tuple(sp.csr_matrix(features.numpy()))
# print(features)

data = {
    'X_original': (features, adj, None),
    'y': labels
}

in_dim = features[2][1]
out_dim = num_labels
n_layers = 4
n_feat_nonzero = features[0].shape[0]
epochs = 200
random_seeds = [123, 345, 234, 683, 372, 385, 348, 823, 644, 765]
test_acc_list = []

for batch in range(10):
    # Set random seeds
    s = random_seeds[batch]
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # get Train_Val_Test Set
    train_mask, val_mask, test_mask = getBatchTrainValTestSet(dataset_str=dataset, batch=batch)
    # SGC
    dropout = 0.0
    gcn_model = SimpleGCNAF(in_dim, out_dim, n_layers, n_feat_nonzero, dropout, device, usePCA=True, useAE=False).to(device)

    # GCN
    # dropout = 0.5
    # dim_list = [in_dim, 16, 16, 16, out_dim]
    # gcn_model = GCN_Model(dim_list=dim_list, n_feat_nonzero=n_feat_nonzero, dropout=dropout, device=device).to(device)

    lr = 0.05
    weight_decay = 1e-4
    optimizer = Adam(gcn_model.parameters(), lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 60, 0.97)
    # start = time.time()
    best_acc, test_acc, train_loss_list, val_loss_list, train_acc_list, val_acc_list = train(gcn_model=gcn_model,
                                                                                             optimizer=optimizer,
                                                                                             scheduler=scheduler,
                                                                                             data=data,
                                                                                             train_mask=train_mask,
                                                                                             val_mask=val_mask,
                                                                                             test_mask=test_mask,
                                                                                             labels_cpu=labels,
                                                                                             epochs=epochs)
    # end = time.time()
    # train_time_list.append(end - start)
    # test_time = getTestTime(gcn_model, data, labels_cpu, test_mask)
    # test_time_list.append(test_time)
    test_acc_list.append(test_acc)
    print('train_loss_list = ', train_loss_list)
    print('val_loss_list = ', val_loss_list)
    print('train_acc_list = ', train_acc_list)
    print('val_acc_list = ', val_acc_list)
print('test_acc_list=', test_acc_list)
print('Average test accuracy:', np.mean(test_acc_list))
