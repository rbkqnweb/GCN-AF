import utils_data


def getAdjMatrix(dataset_str, splits_file_path):
    g, adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = utils_data.load_data(
        dataset_str, splits_file_path=splits_file_path)
    return adj
