from matplotlib import pyplot as plt
from plotting import plot_scatter, visualize_latent_subspaces
from torch.utils.data import DataLoader, TensorDataset
import torch
from model import DSVAE_prior_MNIST
import numpy as np
import yaml
import argparse
from utils import seed_worker, set_seed, counterfactuals

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-step', type=float, default=2)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=64)
    
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_dim = 392 
    data = 'cmnist'
    color = True
    data_path = './data'

    labels = ['4', '9']
    colors = ['red', 'green']

    # Baseline, e=0.5
    e=0.5    
    in_data = str(int(100*e))
    
    dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
    dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
    test_loader_in  = DataLoader(dset_test_in, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    input_path = f'outputs/train_baseline/++seed={args.seed},+bw=10,+bx=1,+by=10,+bz=1,+e={e}'

    csvae = DSVAE_prior_MNIST(x_dim)
    csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
    csvae.load_state_dict(csvae_state)
    csvae = csvae.to(device) 

    plt.rcParams.update({'font.size': 14})

    visualize_latent_subspaces(csvae, test_loader_in, device, f'outputs/exp1_baseline_latent_0.5.png')

    # x_in, y_in, _ = next(iter(test_loader_in))
    # labels_in = [4 if x.item() == 0 else 9 for x in y_in]
    # x_in_CF = counterfactuals(csvae, device, x_in)  

    # f, axarr = plt.subplots(2, 10, figsize=(10, 2), constrained_layout=True)
    # for i, ax in enumerate(axarr[0, :]):
    #     ax.imshow(torch.cat((x_in[i].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xlabel(labels_in[i], fontsize=11)
    # for i, ax in enumerate(axarr[1, :]):
    #     ax.imshow(torch.cat((x_in_CF[i].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
    #     ax.axis('off')
    # plt.savefig(f'outputs/exp1_baseline_cf_0.5.png')
    # plt.close(f)

    fig, axes = plt.subplots(3, 2, figsize=(10, 15), constrained_layout=True)

    # Marginal MMD, e=0.2
    e=0.2    
    in_data = str(int(100*e))
    
    dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
    dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
    test_loader_in  = DataLoader(dset_test_in, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    input_path_mmd = f'outputs/train_mmd/++seed={args.seed},+bmw=10,+bmz=10,+bw=10,+bx=1,+by=10,+bz=1,+conditional=false,+e={e}'
    csvae_mmd = DSVAE_prior_MNIST(x_dim)
    csvae_state_mmd = torch.load(f'{input_path_mmd}/csvae.pt', map_location=device)
    csvae_mmd.load_state_dict(csvae_state_mmd)
    csvae_mmd = csvae_mmd.to(device) 

    legend=True
    with torch.no_grad():
        for x, y, c in test_loader_in:
            x = x.to(device)
            z_mmd = csvae_mmd.posteriorZ(x).sample().cpu()
            if legend:
                for i in range(2):
                    plot_scatter(axes[0, 0],  z_mmd[y == i], [i]*sum(y==i), l=labels[i])
                    plot_scatter(axes[0, 1], z_mmd[c == i], [i]*sum(c==i), l=colors[i], c=['r', 'g'])
                legend=False
            else:
                plot_scatter(axes[0, 0], z_mmd, y)
                plot_scatter(axes[0, 1], z_mmd, c, c=['r', 'g'])

    # Marginal models, e=0.05
    e=0.05    
    in_data = str(int(100*e))
    
    dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
    dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
    test_loader_in  = DataLoader(dset_test_in, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    input_path_mmd = f'outputs/train_mmd/++seed={args.seed},+bmw=10,+bmz=10,+bw=10,+bx=1,+by=10,+bz=1,+conditional=false,+e={e}'
    csvae_mmd = DSVAE_prior_MNIST(x_dim)
    csvae_state_mmd = torch.load(f'{input_path_mmd}/csvae.pt', map_location=device)
    csvae_mmd.load_state_dict(csvae_state_mmd)
    csvae_mmd = csvae_mmd.to(device) 

    input_path_adv = f'outputs/train_adversarial/++seed={args.seed},+bc=1,+bhw=10,+bhz=10,+bw=10,+bx=1,+by=10,+byz=1,+bz=1,+conditional=false,+e={e}'
    csvae_adv = DSVAE_prior_MNIST(x_dim)
    csvae_state_adv = torch.load(f'{input_path_adv}/csvae.pt', map_location=device)
    csvae_adv.load_state_dict(csvae_state_adv)
    csvae_adv = csvae_adv.to(device)

    with torch.no_grad():
        for x, y, c in test_loader_in:
            x = x.to(device)
            z_mmd = csvae_mmd.posteriorZ(x).sample().cpu()
            plot_scatter(axes[1, 0], z_mmd, y)
            plot_scatter(axes[1, 1], z_mmd, c, c=['r', 'g'])
            
            z_adv = csvae_adv.posteriorZ(x).sample().cpu()
            plot_scatter(axes[2, 0], z_adv, y)
            plot_scatter(axes[2, 1], z_adv, c, c=['r', 'g'])

    for i in range(3):
        for j in range(2):
            axes[i, j].set_xlabel('$z_0$')
            axes[i, j].set_ylabel('$z_1$')
    
    for i in range(2):
        axes[0, i].legend()
    
    axes[0, 0].set_title(r'MMD, marginal, $e=0.2$')
    axes[1, 0].set_title(r'MMD, marginal, $e=0.05$')
    axes[2, 0].set_title(r'Adversarial, marginal, $e=0.05$')

    plt.savefig(f'outputs/exp1_marginal_latent.png',  bbox_inches='tight')
    plt.close(fig)