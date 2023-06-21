from matplotlib import pyplot as plt
from plotting import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from model import DSVAE_prior_MNIST, VI_baseline
import argparse
import numpy as np
import yaml
from utils import eval_by
from typing import *
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)

    args = parser.parse_args()

    input_path = args.input

    with open(f"{input_path}/.hydra/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    with open(f"{input_path}/.hydra/hydra.yaml", "r") as f:
        cfg_hydra = yaml.safe_load(f)

    hparams = cfg
    color = hparams['color']
    x_dim = 392 if color else 784
    data = 'mnist' if not color else 'cmnist'
    e_in = hparams['e']
    in_data = str(int(100*e_in))

    n = hparams['n']
    if n != 0:
     in_data += f'_{int(n*100)}'

    x_dim = 392
    data_path = './data'
    dataset_val = torch.load(f'{data_path}/{data}_valid_{in_data}.pt')
    dset_val = TensorDataset(dataset_val['images'], dataset_val['labels'], dataset_val['colors'])
    batch_size = hparams['batch_size']
    val_loader  = DataLoader(dset_val, batch_size=batch_size, shuffle=True)

    torch.manual_seed(hparams["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    csvae = DSVAE_prior_MNIST(x_dim)
    csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
    csvae.load_state_dict(csvae_state)
    csvae = csvae.to(device)

    bx = hparams['bx']
    bw = hparams['bw']
    bz = hparams['bz']
    by = hparams['by']
    
    vi = VI_baseline(bx, bw, bz, by)
    log_px, qy = eval_by(val_loader, csvae, vi, device)

    f = open(f'outputs/by.txt', 'a')
    f.write(f"{by},{log_px:.3f},{qy:.3f}\n")
    f.close()