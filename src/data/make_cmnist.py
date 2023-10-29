import argparse
import torch
from torch.utils.data.dataset import random_split
from torchvision.datasets import MNIST
import numpy as np

def torch_bernoulli(p, size):
  return (torch.rand(size) < p).float()
def torch_xor(a, b):
  return (a-b).abs() # Assumes both inputs are either 0 or 1

def make_environment(images, labels, e, n):
  # 2x subsample for computational convenience
  images = images.reshape((-1, 28, 28))[:, ::2, ::2]
  # Label 4 as 0 and 9 as 1
  labels = (labels == 9).float()
  # Flip label with probability n
  labels = torch_xor(labels, torch_bernoulli(n, len(labels)))
  # Assign a color based on the label; flip the color with probability e
  colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
  # Apply the color to the image by zeroing out the other color channel
  images = torch.stack([images, images], dim=1)
  images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
  return {
    'images': images.view(-1, 392).float()/255., 
    'labels': labels, 
    'colors': colors
  }

def make_cmnist(data_path, e, seed, n):
  """
  Make CMNIST dataset
  """
  # Download MNIST dataset
  dset_train = MNIST(data_path, train=True,  download=True)
  dset_test  = MNIST(data_path, train=False, download=True)

  # Find index of images which are labeled 4 or 9
  classes = torch.tensor([4, 9])
  train_index = (dset_train.targets[:, None] == classes).any(-1).nonzero(as_tuple=True)[0]
  test_index = (dset_test.targets[:, None] == classes).any(-1).nonzero(as_tuple=True)[0]

  # Split the subset of MNIST with only 4 and 9 into train, validation, and test sets
  n_test = len(test_index)
  n_train = len(train_index) - n_test 
  generator = torch.Generator().manual_seed(seed)
  mnist_train = (dset_train.data[train_index][:n_train], dset_train.targets[train_index][:n_train])
  mnist_valid = (dset_train.data[train_index][n_train:], dset_train.targets[train_index][n_train:])
  mnist_test = (dset_test.data[test_index], dset_test.targets[test_index])

  rng_state = np.random.get_state(seed)
  np.random.shuffle(mnist_train[0].numpy())
  np.random.set_state(rng_state)
  np.random.shuffle(mnist_train[1].numpy()) 

  cmnist_train = make_environment(mnist_train[0], mnist_train[1], e, n)
  cmnist_valid = make_environment(mnist_valid[0], mnist_valid[1], e, n)
  cmnist_test = make_environment(mnist_test[0], mnist_test[1], e, n)

  in_data = str(int(e*100))
  out_data =  str(int((1-e)*100))
  if n != 0:
    in_data += f'_{int(n*100)}'
    out_data += f'_{int(n*100)}'

  torch.save(cmnist_train, f'{data_path}/cmnist_train_{in_data}.pt')
  torch.save(cmnist_valid, f'{data_path}/cmnist_valid_{in_data}.pt')
  torch.save(cmnist_test, f'{data_path}/cmnist_test_{in_data}.pt')

  # OOD test data
  cmnist_test = make_environment(mnist_test[0], mnist_test[1], 1-e, n)
  torch.save(cmnist_test, f'{data_path}/cmnist_test_{out_data}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=float)
    parser.add_argument('-n', type=float, default=0)
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    data_path = '../data'
    make_cmnist(data_path, args.e, args.seed, args.n)