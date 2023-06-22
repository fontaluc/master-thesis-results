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
    parser.add_argument('-quanti', type=bool, default=True)
    parser.add_argument('-quali', type=bool, default=True)

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

    if args.quanti:
        bx = hparams['bx']
        bw = hparams['bw']
        bz = hparams['bz']
        by = hparams['by']
        
        vi = VI_baseline(bx, bw, bz, by)
        log_px, qy = eval_by(val_loader, csvae, vi, device)

        lr = hparams['lr']
        if not os.path.exists(f'{input_path}/adv_y_cond_nl.pt'):
            aux_y_cond, aux_c_cond = train_cond_adversary(nl_adversary, csvae, lr, device, train_loader, val_loader, input_path)
        else:
            aux_y_cond = load_adversary(nl_adversary, 3, f'{input_path}/adv_y_cond_nl.pt', device)
            aux_c_cond = load_adversary(nl_adversary, 3, f'{input_path}/adv_c_cond_nl.pt', device)
        acc_y_cond, acc_c_cond = eval_cond_adversary(csvae, aux_y_cond, aux_c_cond, val_loader, device)

        if not os.path.exists(f'{input_path}/adv_y_marg_nl.pt'):
            aux_y_marg, aux_c_marg = train_marg_adversary(nl_adversary, csvae, lr, device, train_loader, val_loader, input_path)
        else:
            aux_y_marg = load_adversary(nl_adversary, 2, f'{input_path}/adv_y_marg_nl.pt', device)
            aux_c_marg = load_adversary(nl_adversary, 2, f'{input_path}/adv_c_marg_nl.pt', device)
        acc_y_marg, acc_c_marg = eval_marg_adversary(csvae, aux_y_marg, aux_c_marg, val_loader, device)

        if not os.path.exists(f'{input_path}/adv_y0_nl.pt'):
            aux_y0, aux_c0, aux_y1, aux_c1 = train_adversary_dual(nl_adversary, csvae, lr, device, train_loader, val_loader, input_path)
        else:
            aux_y0 = load_adversary(nl_adversary, 2, f'{input_path}/adv_y0_nl.pt', device)
            aux_c0 = load_adversary(nl_adversary, 2, f'{input_path}/adv_c0_nl.pt', device)
            aux_y1 = load_adversary(nl_adversary, 2, f'{input_path}/adv_y1_nl.pt', device)
            aux_c1 = load_adversary(nl_adversary, 2, f'{input_path}/adv_c1_nl.pt', device)
        acc_y_dual, acc_c_dual = eval_adversary_dual(csvae, aux_y0, aux_c0, aux_y1, aux_c1, val_loader, device)

        mmd_w0, mmd_w1, mmd_z0, mmd_z1 = get_conditional_mmd(dataset_val, csvae, device)
        mmd_w = (mmd_w0 + mmd_w1)/2
        mmd_z = (mmd_z0 + mmd_z1)/2
        n_switch, n_appear = get_color_switching_ratio(val_loader, csvae, device, step=args.step)

        if 'bmw' in hparams.keys():
            bm = hparams['bmw']
        elif 'bhw' in hparams.keys():
            bm = hparams['bhw']
        else:
            bm = 0
        
        # model = cfg_hydra['hydra']['job']['name'][6:]
        # f = open(f'outputs/bm_bis2.txt', 'a')
        # f.write(f"{model},{bm},{log_px:.3f},{qy:.3f},{acc_y_marg:.3f},{acc_c_marg:.3f},{acc_y_cond:.3f},{acc_c_cond:.3f},{acc_y_dual:.3f},{acc_c_dual:.3f},{mmd_w:.3f},{mmd_z:.3f},{n_appear}\n")
        # f.close()

        x_train = dataset_train['images']
        y_train = dataset_train['labels']
        x_val = dataset_val['images']
        y_val = dataset_val['labels']
        c_train = dataset_train['colors']
        c_val = dataset_val['colors']

        with torch.no_grad():
            outputs_train = csvae(x_train.to(device), y_train.to(device))
            outputs_val = csvae(x_val.to(device), y_val.to(device))

        w_train, z_train = [outputs_train[k] for k in ["w", "z"]]
        w_val, z_val = [outputs_val[k] for k in ["w", "z"]]

        # Decision boundary with the train and validation data points
        z0_train = z_train[c_train == 0]
        z0_val = z_val[c_val == 0]
        y0_train = y_train[c_train == 0]
        y0_val = y_val[c_val == 0]
        z1_train = z_train[c_train == 1]
        z1_val = z_val[c_val == 1]
        y1_train = y_train[c_train == 1]
        y1_val = y_val[c_val == 1]
        w0_train = w_train[y_train == 0]
        w0_val = w_val[y_val == 0]
        c0_train = c_train[y_train == 0]
        c0_val = c_val[y_val == 0]
        w1_train = w_train[y_train == 1]
        w1_val = w_val[y_val == 1]
        c1_train = c_train[y_train == 1]
        c1_val = c_val[y_val == 1]

        visualize_decision_boundary(predict(aux_y0), [z0_train, z0_val], [y0_train, y0_val], ['$z_1$', '$z_2$'], ['4', '9'], f"{input_path}/decision_boundary_y0.png")
        visualize_decision_boundary(predict(aux_c0), [w0_train, w0_val], [c0_train, c0_val], ['$w_1$', '$w_2$'], ['red', 'green'], f"{input_path}/decision_boundary_c0.png", ['r', 'g'])
        visualize_decision_boundary(predict(aux_y1), [z1_train, z1_val], [y1_train, y1_val], ['$z_1$', '$z_2$'], ['4', '9'], f"{input_path}/decision_boundary_y1.png")
        visualize_decision_boundary(predict(aux_c1), [w1_train, w1_val], [c1_train, c1_val], ['$w_1$', '$w_2$'], ['red', 'green'], f"{input_path}/decision_boundary_c1.png", ['r', 'g'])

        visualize_decision_boundary(predict(aux_y_marg), [z_train, z_val], [y_train, y_val], ['$z_1$', '$z_2$'], ['4', '9'], f"{input_path}/decision_boundary_y.png")
        visualize_decision_boundary(predict(aux_c_marg), [w_train, w_val], [c_train, c_val], ['$w_1$', '$w_2$'], ['red', 'green'], f"{input_path}/decision_boundary_c.png", ['r', 'g'])

    if args.quali:
        visualize_latent_subspaces(csvae, val_loader, device, f"{input_path}/latent-valid.png")
        visualize_label_counterfactuals(val_loader, csvae, device, input_path, 'valid', color=color, step=args.step)