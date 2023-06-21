import torch
from torch.utils.data import TensorDataset
dataset_train = torch.load('data/cmnist_train_20.pt')
# dataset_test = torch.load('data/cmnist_test_20.pt')
# print(len(dataset_train['images']), len(dataset_test['images']))

dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
print(dset_train[:, 0])
