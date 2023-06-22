from matplotlib import pyplot as plt
from plotting import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from model import DSVAE_prior_MNIST, nl_adversary, l_adversary
import argparse
import numpy as np
import yaml
from utils import *
from typing import *
from sklearn.metrics import accuracy_score
import os

def predict(aux):
    return lambda x: F.softmax(aux(torch.tensor(x, dtype=torch.float)), dim=1)[:, 1].data.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-step', type=float, default=2)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-lr', type=float, default=0.005)
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-type', type=str, choices=['marg', 'cond', 'dual'])
    parser.add_argument('-linear', type=bool, default=False)

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

    x_dim = 392
    data_path = './data'
    dataset_train = torch.load(f'{data_path}/{data}_train_{in_data}.pt')
    dataset_val = torch.load(f'{data_path}/{data}_valid_{in_data}.pt')
    dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
    dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
    dset_val = TensorDataset(dataset_val['images'], dataset_val['labels'], dataset_val['colors'])
    dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
    batch_size = hparams['batch_size']
    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(dset_val, batch_size=batch_size, shuffle=True)
    test_loader_in  = DataLoader(dset_test_in, batch_size=batch_size, shuffle=True)

    torch.manual_seed(hparams["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    csvae = DSVAE_prior_MNIST(x_dim)
    csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
    csvae.load_state_dict(csvae_state)
    csvae = csvae.to(device)

    lr = hparams['lr']
    adversary = l_adversary if args.linear else nl_adversary
    num_input = 3 if args.type == 'cond' else 2

    if args.train:
        if args.type == 'cond':
            aux_y, aux_c = train_cond_adversary(adversary, csvae, lr, device, train_loader, val_loader, input_path, args.epochs)
        elif args.type == 'marg':
            aux_y, aux_c = train_marg_adversary(adversary, csvae, lr, device, train_loader, val_loader, input_path, args.epochs)
        else:
            aux_y0, aux_c0, aux_y1, aux_c1 = train_adversary_dual(adversary, csvae, lr, device, train_loader, val_loader, input_path, args.epochs)
    else:
        if args.type == 'dual':
            aux_y0 = load_adversary(adversary, num_input, f'{input_path}/adv_y0_{args.type}_nl.pt', device)
            aux_c0 = load_adversary(adversary, num_input, f'{input_path}/adv_c0_{args.type}_nl.pt', device)
            aux_y1 = load_adversary(adversary, num_input, f'{input_path}/adv_y1_{args.type}_nl.pt', device)
            aux_c1 = load_adversary(adversary, num_input, f'{input_path}/adv_c1_{args.type}_nl.pt', device)
        else:
            aux_y = load_adversary(adversary, num_input, f'{input_path}/adv_y_{args.type}_nl.pt', device)
            aux_c = load_adversary(adversary, num_input, f'{input_path}/adv_c_{args.type}_nl.pt', device)
    
    if args.type == 'cond':
        acc_y, acc_c = eval_cond_adversary(csvae, aux_y, aux_c, test_loader_in, device)
    elif args.type == 'marg':
        acc_y, acc_c = eval_marg_adversary(csvae, aux_y, aux_c, test_loader_in, device)
    else:
        acc_y, acc_c = eval_adversary_dual(csvae, aux_y0, aux_c0, aux_y1, aux_c1, test_loader_in, device)

    model = cfg_hydra['hydra']['job']['name']
    conditional = False if model == 'train_baseline' else hparams['conditional']
    tag = 'l' if args.linear else 'nl'
    f = open(f'outputs/{args.type}_{tag}_adv_acc.txt', 'a')
    f.write(f"{model}, {conditional}, {e_in}, {acc_y:.3f}, {acc_c:.3f}\n")
    f.close()

    if args.type != 'cond':
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

        if args.type == 'dual':
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

            visualize_decision_boundary(predict(aux_y0), [z0_train, z0_val], [y0_train, y0_val], ['$z_1$', '$z_2$'], ['4', '9'], f"{input_path}/decision_boundary_y0_{tag}.png")
            visualize_decision_boundary(predict(aux_c0), [w0_train, w0_val], [c0_train, c0_val], ['$w_1$', '$w_2$'], ['red', 'green'], f"{input_path}/decision_boundary_c0_{tag}.png", ['r', 'g'])
            visualize_decision_boundary(predict(aux_y1), [z1_train, z1_val], [y1_train, y1_val], ['$z_1$', '$z_2$'], ['4', '9'], f"{input_path}/decision_boundary_y1_{tag}.png")
            visualize_decision_boundary(predict(aux_c1), [w1_train, w1_val], [c1_train, c1_val], ['$w_1$', '$w_2$'], ['red', 'green'], f"{input_path}/decision_boundary_c1_{tag}.png", ['r', 'g'])

        else:
            visualize_decision_boundary(predict(aux_y), [z_train, z_val], [y_train, y_val], ['$z_1$', '$z_2$'], ['4', '9'], f"{input_path}/decision_boundary_y_{tag}.png")
            visualize_decision_boundary(predict(aux_c), [w_train, w_val], [c_train, c_val], ['$w_1$', '$w_2$'], ['red', 'green'], f"{input_path}/decision_boundary_c_{tag}.png", ['r', 'g'])