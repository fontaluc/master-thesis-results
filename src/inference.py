from matplotlib import pyplot as plt
from plotting import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from model import DSVAE_prior_MNIST, nl_adversary
import argparse
import numpy as np
import yaml
from utils import *
from typing import *
from sklearn.metrics import accuracy_score

def predict_cat(X, decoderY):
    X = torch.from_numpy(X).float()
    logits = decoderY(X)
    return torch.argmax(logits, 1).data.numpy()

def predict_bern(X, decoderY):
    X = torch.from_numpy(X).float()
    probs = decoderY(X)
    return (probs > 0.5).int().data.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-step', type=float, default=2)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-train_marg', type=bool, default=False)
    parser.add_argument('-train_cond', type=bool, default=False)
    parser.add_argument('-visualize', type=bool, default=False)

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
    e_out = 1 - e_in
    in_data = str(int(100*e_in))
    out_data = str(int(100*e_out))
    bx = hparams['bx']

    if 'n' in hparams.keys():
        n = hparams['n']
        if n != 0:
            in_data += f'_{int(n*100)}'
            out_data += f'_{int(n*100)}'

    x_dim = 392
    data_path = './data'
    dataset_train = torch.load(f'{data_path}/{data}_train_{in_data}.pt')
    dataset_val = torch.load(f'{data_path}/{data}_valid_{in_data}.pt')
    dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
    dataset_test_out = torch.load(f'{data_path}/{data}_test_{out_data}.pt')
    # dataset_train = torch.load(f'{data_path}/{data}_{in_data}_train.pt')
    # dataset_val = torch.load(f'{data_path}/{data}_{in_data}_valid.pt')
    # dataset_test_in = torch.load(f'{data_path}/{data}_{in_data}_test.pt')
    # dataset_test_out = torch.load(f'{data_path}/{data}_{out_data}_test.pt')
    dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
    dset_val = TensorDataset(dataset_val['images'], dataset_val['labels'], dataset_val['colors'])
    dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
    dset_test_out = TensorDataset(dataset_test_out['images'], dataset_test_out['labels'], dataset_test_out['colors'])
    batch_size = hparams['batch_size']
    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(dset_val, batch_size=batch_size, shuffle=True)
    test_loader_in  = DataLoader(dset_test_in, batch_size=batch_size, shuffle=True)
    test_loader_out  = DataLoader(dset_test_out, batch_size=batch_size, shuffle=True)

    torch.manual_seed(hparams["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    csvae = DSVAE_prior_MNIST(x_dim)
    csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
    csvae.load_state_dict(csvae_state)
    csvae = csvae.to(device)

    lr = hparams['lr']

    if args.train:
        nl_aux_y, nl_aux_c = train_adversary(nl_adversary, csvae, lr, device, train_loader, val_loader, test_loader_in, input_path, 'nl', args.epochs)
    else:
        nl_aux_y = nl_adversary(2)
        nl_aux_y_state = torch.load(f'{input_path}/adv_y_nl.pt', map_location=device)
        nl_aux_y.load_state_dict(nl_aux_y_state)
        nl_aux_y = nl_aux_y.to(device)

        nl_aux_c = nl_adversary(2)
        nl_aux_c_state = torch.load(f'{input_path}/adv_c_nl.pt', map_location=device)
        nl_aux_c.load_state_dict(nl_aux_c_state)
        nl_aux_c = nl_aux_c.to(device)

    acc_y, acc_c  = eval_adversary(csvae, nl_aux_y, nl_aux_c, test_loader_in, device)
    acc_in = get_cmnist_accuracy(test_loader_in, csvae.classifier, device)
    acc_out = get_cmnist_accuracy(test_loader_out, csvae.classifier, device)
    mmd_w0, mmd_w1, mmd_z0, mmd_z1 = get_conditional_mmd(dataset_test_in, csvae, device)
    n_switch, n_appear = get_color_switching_ratio(test_loader_in, csvae, device, step=args.step)
    model = cfg_hydra['hydra']['job']['name']
    conditional = False if model == 'train_baseline' else hparams['conditional']
    if bx != 1:
        in_data += f'_{bx}'
    f = open(f'outputs/exp1.txt', 'a')
    f.write(f"{model}, {conditional}, {e_in}, {acc_in:.3f}, {acc_out:.3f}, {acc_y:.3f}, {acc_c:.3f}, {(mmd_w0 + mmd_w1)/2:.3f}, {(mmd_z0 + mmd_z1)/2:.3f}, {n_appear}\n")
    f.close()

    if args.visualize:
        visualize_latent_subspaces(csvae, test_loader_in, device, f"{input_path}/latent-test-in.png")
        visualize_label_counterfactuals(test_loader_in, csvae, device, input_path, 'test', color=color, step=args.step)