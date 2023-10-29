from matplotlib import pyplot as plt
from src.plotting import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from src.models.model import DSVAE_prior_MNIST, nl_adversary
import argparse
import numpy as np
import yaml
from src.utils import *
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
    parser.add_argument('-visualize', type=bool, default=True)
    parser.add_argument('-ood', action='store_true')

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
    e_out = 1 - e_in
    in_data = str(int(100*e_in))
    out_data = str(int(100*e_out))
    bx = hparams['bx']
    bw = hparams['bw']
    by = hparams['by']

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
    dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
    dset_val = TensorDataset(dataset_val['images'], dataset_val['labels'], dataset_val['colors'])
    dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
    dset_test_out = TensorDataset(dataset_test_out['images'], dataset_test_out['labels'], dataset_test_out['colors'])
    batch_size = hparams['batch_size']
    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader  = DataLoader(dset_val, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader_in  = DataLoader(dset_test_in, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader_out  = DataLoader(dset_test_out, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    torch.manual_seed(hparams["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    m0 = hparams['m0']
    s0 = hparams['s0']
    m1 = hparams['m1']
    s1 = hparams['s1']
    conv = hparams['conv'] if 'conv' in hparams.keys() else False
    csvae = DSVAE_prior_MNIST(x_dim=x_dim, m0=m0, s0=s0, m1=m1, s1=s1) if not conv else DSCVAE_prior_MNIST(m0=m0, s0=s0, m1=m1, s1=s1)
    csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
    csvae.load_state_dict(csvae_state)
    csvae = csvae.to(device)

    lr = hparams['lr']
    cpu = torch.device('cpu')

    if not os.path.exists(f'{input_path}/adv_y_cond_nl.pt'):
        aux_y_cond, aux_c_cond = train_cond_adversary(nl_adversary, csvae, lr, device, train_loader, val_loader, input_path)
        aux_y_cond.to(cpu)
        aux_c_cond.to(cpu)
    else:
        aux_y_cond = load_adversary(nl_adversary, 3, f'{input_path}/adv_y_cond_nl.pt', cpu)
        aux_c_cond = load_adversary(nl_adversary, 3, f'{input_path}/adv_c_cond_nl.pt', cpu)
    acc_y_cond_in, acc_c_cond_in = eval_cond_adversary(csvae, aux_y_cond, aux_c_cond, test_loader_in, cpu)

    if not os.path.exists(f'{input_path}/adv_y_marg_nl.pt'):
        aux_y_marg, aux_c_marg = train_marg_adversary(nl_adversary, csvae, lr, device, train_loader, val_loader, input_path)
        aux_y_marg.to(cpu)
        aux_c_marg.to(cpu)
    else:
        aux_y_marg = load_adversary(nl_adversary, 2, f'{input_path}/adv_y_marg_nl.pt', cpu)
        aux_c_marg = load_adversary(nl_adversary, 2, f'{input_path}/adv_c_marg_nl.pt', cpu)
    acc_y_marg_in, acc_c_marg_in = eval_marg_adversary(csvae, aux_y_marg, aux_c_marg, test_loader_in, cpu)

    acc_in = get_cmnist_accuracy(test_loader_in, csvae.classifier, device)
    acc_out = get_cmnist_accuracy(test_loader_out, csvae.classifier, device)
    mmd_w_in, mmd_z_in = get_conditional_mmd(dataset_test_in, csvae, device)
    n_appear_in = get_color_switching_ratio(test_loader_in, csvae, device, step=args.step)
    model = cfg_hydra['hydra']['job']['name']
    conditional = hparams['conditional'] if model in ['train_mmd', 'train_adversarial'] else None
    if bx != 1:
        in_data += f'_{bx}'
    if 'n' in hparams.keys() and hparams['n'] == 0.2:
        f = open(f'outputs/results/exp2_bw={bw}_by={by}.txt', 'a')
        f.write(f"{model},{conditional},{acc_in:.3f},{acc_out:.3f},{acc_y_marg_in:.3f},{acc_c_marg_in:.3f},{acc_y_cond_in:.3f},{acc_c_cond_in:.3f},{mmd_w_in:.3f},{mmd_z_in:.3f},{n_appear_in}\n")
        f.close()

        if args.ood:

            acc_y_cond_out, acc_c_cond_out = eval_cond_adversary(csvae, aux_y_cond, aux_c_cond, test_loader_out, cpu)
            acc_y_marg_out, acc_c_marg_out = eval_marg_adversary(csvae, aux_y_marg, aux_c_marg, test_loader_out, cpu)
            mmd_w_out, mmd_z_out = get_conditional_mmd(dataset_test_out, csvae, device)
            n_appear_out = get_color_switching_ratio(test_loader_out, csvae, device, step=args.step)

            f = open(f'outputs/results/exp2_bw={bw}_by={by}_iid.txt', 'a')
            f.write(f"{model},{conditional},{acc_in:.3f},{acc_y_marg_in:.3f},{acc_c_marg_in:.3f},{acc_y_cond_in:.3f},{acc_c_cond_in:.3f},{mmd_w_in:.3f},{mmd_z_in:.3f},{n_appear_in}\n")
            f.close()

            f = open(f'outputs/results/exp2_bw={bw}_by={by}_ood.txt', 'a')
            f.write(f"{model},{conditional},{acc_out:.3f},{acc_y_marg_out:.3f},{acc_c_marg_out:.3f},{acc_y_cond_out:.3f},{acc_c_cond_out:.3f},{mmd_w_out:.3f},{mmd_z_out:.3f},{n_appear_out}\n")
            f.close()
    else:
        f = open(f'outputs/results/exp1.txt', 'a')
        f.write(f"{model},{conditional},{e_in},{acc_in:.3f},{acc_out:.3f},{acc_y_marg_in:.3f},{acc_c_marg_in:.3f},{acc_y_cond_in:.3f},{acc_c_cond_in:.3f},{mmd_w_in:.3f},{mmd_z_in:.3f},{n_appear_in}\n")
        f.close()

    if args.visualize:
        visualize_latent_subspaces(csvae, test_loader_in, device, f"{input_path}/latent-test-in.png")
        visualize_label_counterfactuals(test_loader_in, csvae, device, f"{input_path}/counterfactuals-test-in.png", color=color, step=args.step)
        visualize_latent_subspaces(csvae, test_loader_out, device, f"{input_path}/latent-test-out.png")
        visualize_label_counterfactuals(test_loader_out, csvae, device, f"{input_path}/counterfactuals-test-out.png", color=color, step=args.step)