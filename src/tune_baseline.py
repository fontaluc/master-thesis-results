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
from utils import eval_by, eval_bw, seed_worker, set_seed
from typing import *
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-bw', action='store_true')
    parser.add_argument('-by', action='store_true')

    args = parser.parse_args()

    input_path = args.input

    with open(f"{input_path}/.hydra/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    with open(f"{input_path}/.hydra/hydra.yaml", "r") as f:
        cfg_hydra = yaml.safe_load(f)

    hparams = cfg

    set_seed(hparams["seed"])
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(hparams["seed"])

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

    dataset_train = torch.load(f'{data_path}/{data}_train_{in_data}.pt')
    dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
    batch_size = hparams['batch_size']
    train_loader  = DataLoader(dset_train, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    dataset_val = torch.load(f'{data_path}/{data}_valid_{in_data}.pt')
    dset_val = TensorDataset(dataset_val['images'], dataset_val['labels'], dataset_val['colors'])
    batch_size = hparams['batch_size']
    val_loader  = DataLoader(dset_val, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

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

    if args.bw:
        log_px, kl_w, dkl_w = eval_bw(train_loader, val_loader, csvae, vi, device)
        f = open(f'outputs/bw.txt', 'a')
        f.write(f"{bw},{log_px:.3f},{kl_w:.3f},{dkl_w:.3f}\n")
        f.close()

    if args.by:
        _, qy_train = eval_by(train_loader, csvae, vi, device)
        log_px_val, qy_val = eval_by(val_loader, csvae, vi, device)
        f = open(f'outputs/by_bw={bw}.txt', 'a')
        f.write(f"{by},{log_px:.3f},{qy_train:.3f},{qy_val:.3f}\n")
        f.close()

        visualize_latent_subspaces(csvae, val_loader, device, f"{input_path}/latent-valid.png")
        visualize_latent_subspaces(csvae, train_loader, device, f"{input_path}/latent-train.png")
    
