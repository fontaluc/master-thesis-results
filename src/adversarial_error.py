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
    parser.add_argument('-cond', type=bool, default=True)

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

    if args.train:
        if args.cond:
            nl_aux_y, nl_aux_c = train_cond_adversary(nl_adversary, csvae, lr, device, train_loader, val_loader, input_path, 'nl', args.epochs)
            l_aux_y, l_aux_c = train_cond_adversary(l_adversary, csvae, lr, device, train_loader, val_loader, input_path, 'l', args.epochs)
        else:
            nl_aux_y, nl_aux_c = train_marg_adversary(nl_adversary, csvae, lr, device, train_loader, val_loader, input_path, 'nl', args.epochs)
            l_aux_y, l_aux_c = train_marg_adversary(l_adversary, csvae, lr, device, train_loader, val_loader, input_path, 'l', args.epochs)
    else:
        nl_aux_y = nl_adversary(2)
        nl_aux_y_state = torch.load(f'{input_path}/adv_y_nl.pt', map_location=device)
        nl_aux_y.load_state_dict(nl_aux_y_state)
        nl_aux_y = nl_aux_y.to(device)

        nl_aux_c = nl_adversary(2)
        nl_aux_c_state = torch.load(f'{input_path}/adv_c_nl.pt', map_location=device)
        nl_aux_c.load_state_dict(nl_aux_c_state)
        nl_aux_c = nl_aux_c.to(device)

        l_aux_y = l_adversary(2)
        l_aux_y_state = torch.load(f'{input_path}/adv_y_l.pt', map_location=device)
        l_aux_y.load_state_dict(l_aux_y_state)
        l_aux_y = l_aux_y.to(device)

        l_aux_c = l_adversary(2)
        l_aux_c_state = torch.load(f'{input_path}/adv_c_l.pt', map_location=device)
        l_aux_c.load_state_dict(l_aux_c_state)
        l_aux_c = l_aux_c.to(device)
    
    if args.cond:
        l_acc_y, l_acc_c = eval_cond_adversary(csvae, l_aux_y, l_aux_c, test_loader_in, device)
        nl_acc_y, nl_acc_c = eval_cond_adversary(csvae, nl_aux_y, nl_aux_c, test_loader_in, device)
    else:
        l_acc_y, l_acc_c = eval_marg_adversary(csvae, l_aux_y, l_aux_c, test_loader_in, device)
        nl_acc_y, nl_acc_c = eval_marg_adversary(csvae, nl_aux_y, nl_aux_c, test_loader_in, device)
    model = cfg_hydra['hydra']['job']['name']
    conditional = False if model == 'train_baseline' else hparams['conditional']
    tag = 'cond' if args.cond else 'marg'
    f = open(f'outputs/{tag}_adversarial_acc.txt', 'a')
    f.write(f"{model}, {conditional}, {e_in}, {l_acc_y:.3f}, {l_acc_c:.3f}, {nl_acc_y:.3f}, {nl_acc_c:.3f}\n")
    f.close()

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
    visualize_decision_boundary(predict(nl_aux_y), [z_train, z_val], [y_train, y_val], ['$z_1$', '$z_2$'], ['4', '9'], f"{input_path}/decision_boundary_y_nl.png")
    visualize_decision_boundary(predict(nl_aux_c), [w_train, w_val], [c_train, c_val], ['$w_1$', '$w_2$'], ['red', 'green'], f"{input_path}/decision_boundary_c_nl.png", ['r', 'g'])

    visualize_decision_boundary(predict(l_aux_y), [z_train, z_val], [y_train, y_val], ['$z_1$', '$z_2$'], ['4', '9'], f"{input_path}/decision_boundary_y_l.png")
    visualize_decision_boundary(predict(l_aux_c), [w_train, w_val], [c_train, c_val], ['$w_1$', '$w_2$'], ['red', 'green'], f"{input_path}/decision_boundary_c_l.png", ['r', 'g'])