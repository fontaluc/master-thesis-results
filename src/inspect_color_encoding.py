from plotting import plot_scatter, plot_samples, counterfactual_projection
from model import DSVAE_prior_MNIST
from torch.utils.data import DataLoader, TensorDataset
import torch 
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-batch_size', type=int, default=64)

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
    dataset_val = torch.load(f'{data_path}/{data}_valid_{in_data}.pt')
    dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
    dset_val = TensorDataset(dataset_val['images'], dataset_val['labels'], dataset_val['colors'])
    batch_size = hparams['batch_size']
    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(dset_val, batch_size=batch_size, shuffle=True)

    torch.manual_seed(hparams["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    csvae = DSVAE_prior_MNIST(x_dim)
    csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
    csvae.load_state_dict(csvae_state)
    csvae = csvae.to(device)

    x, y, c = next(iter(val_loader))
    w = csvae.posteriorW(x).sample()
    z = csvae.posteriorZ(x).sample()
    perm = torch.cat((torch.arange(196, 392), torch.arange(0, 196)), dim=0)
    x_CCF = x[:, perm]
    w_CCF = csvae.posteriorW(x_CCF).sample()
    # z_CCF = csvae.posteriorZ(x_CCF).sample()

    # x_CCF_rec = csvae.decode(w_CCF, z_CCF)
    # x_rec = csvae.decode(w, z)

    # fig, axes = plt.subplots(2, 2)
    # plot_samples(axes[0, 0], x, color=True)
    # plot_samples(axes[0, 1], x_rec, color=True)
    # plot_samples(axes[1, 0], x_CCF, color=True)
    # plot_samples(axes[1, 1], x_CCF_rec, color=True)
    # axes[0, 0].set_title('Original')
    # axes[0, 1].set_title('Reconstruction')
    # plt.savefig(f'{input_path}/reconstructions_valid.png')
    # plt.close(fig)

    dw = w_CCF - w
    fig, ax = plt.subplots()
    plot_scatter(ax, w, y)
    for i in range(batch_size):
        if y[i] == 0:
            ax.arrow(w[i, 0], w[i, 1], dw[i, 0], dw[i, 1], head_width=0.1)
    plt.savefig(f'{input_path}/flip_color.png')
    plt.close(fig)

    # counterfactual_projection(train_loader, val_loader, 0, csvae, 2, input_path)