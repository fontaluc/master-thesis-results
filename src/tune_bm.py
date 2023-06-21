from matplotlib import pyplot as plt
from plotting import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from model import DSVAE_prior_MNIST, VI_baseline, nl_adversary
import argparse
import numpy as np
import yaml
from utils import *
from typing import *
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-step', type=float, default=2)

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

    dataset_train = torch.load(f'{data_path}/{data}_train_{in_data}.pt')
    dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
    batch_size = hparams['batch_size']
    train_loader  = DataLoader(dset_train, batch_size=batch_size, shuffle=True)

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

    # bx = hparams['bx']
    # bw = hparams['bw']
    # bz = hparams['bz']
    # by = hparams['by']
    
    # vi = VI_baseline(bx, bw, bz, by)
    # log_px, qy = eval_by(val_loader, csvae, vi, device)

    # lr = hparams['lr']
    # aux_y_cond, aux_c_cond = train_cond_adversary(nl_adversary, csvae, lr, device, train_loader, val_loader, input_path)
    # acc_y_cond, acc_c_cond = eval_cond_adversary(csvae, aux_y_cond, aux_c_cond, val_loader, device)

    # aux_y_marg, aux_c_marg = train_marg_adversary(nl_adversary, csvae, lr, device, train_loader, val_loader, input_path)
    # acc_y_marg, acc_c_marg = eval_marg_adversary(csvae, aux_y_marg, aux_c_marg, val_loader, device)

    # mmd_w0, mmd_w1, mmd_z0, mmd_z1 = get_conditional_mmd(dataset_val, csvae, device)
    # mmd_w = (mmd_w0 + mmd_w1)/2
    # mmd_z = (mmd_z0 + mmd_z1)/2
    # n_switch, n_appear = get_color_switching_ratio(val_loader, csvae, device, step=args.step)

    # if 'bmw' in hparams.keys():
    #     bm = hparams['bmw']
    # elif 'bhw' in hparams.keys():
    #     bm = hparams['bhw']
    # else:
    #     bm = 0
    
    # model = cfg_hydra['hydra']['job']['name'][6:]
    # f = open(f'outputs/bm_bis.txt', 'a')
    # f.write(f"{model},{bm},{log_px:.3f},{qy:.3f},{acc_y_marg:.3f},{acc_c_marg:.3f},{acc_y_cond:.3f},{acc_c_cond:.3f},{mmd_w:.3f},{mmd_z:.3f},{n_appear}\n")
    # f.close()

    visualize_latent_subspaces(csvae, val_loader, device, f"{input_path}/latent-valid.png")
    visualize_label_counterfactuals(val_loader, csvae, device, input_path, 'valid', color=color, step=args.step)