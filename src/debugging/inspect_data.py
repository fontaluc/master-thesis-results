import torch
from torch.utils.data import TensorDataset, DataLoader
dataset_train = torch.load('data/cmnist_train_20.pt')
# dataset_test = torch.load('data/cmnist_test_20.pt')
# print(len(dataset_train['images']), len(dataset_test['images']))

dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
train_loader  = DataLoader(dset_train, batch_size=10, shuffle=True)
x, y, c = next(iter(train_loader))
labels = [4 if x.item() == 0 else 9 for x in y]
print(labels)
