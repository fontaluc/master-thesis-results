from matplotlib import pyplot as plt
from src.plotting import plot_scatter, visualize_latent_subspaces
from torch.utils.data import DataLoader, TensorDataset
import torch
from src.models.model import DSVAE_prior_MNIST
import numpy as np
import yaml
import argparse
from src.utils import seed_worker, set_seed, counterfactuals

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
    data_path = '../data'

    e=0.2    
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
    # visualize_latent_subspaces(csvae, test_loader_in, device, f'outputs/exp1_baseline_latent_{e}.png')

    # x_in, y_in, _ = next(iter(test_loader_in))
    # labels_in = [4 if x.item() == 0 else 9 for x in y_in]
    # x_in_CF = counterfactuals(csvae, device, x_in)  

    # f, axarr = plt.subplots(10, 2, figsize=(2, 10), constrained_layout=True)
    # for i, ax in enumerate(axarr[:, 0]):
    #     ax.imshow(torch.cat((x_in[i].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xlabel(labels_in[i], fontsize=11)
    # for i, ax in enumerate(axarr[:, 1]):
    #     ax.imshow(torch.cat((x_in_CF[i].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
    #     ax.axis('off')
    # plt.savefig(f'outputs/exp1_baseline_cf_{e}.png')
    # plt.close(f)

    labels = ['$y=0$', "$y=1$"]
    colors = ['$c=0$', '$c=1$']
    fig, axes = plt.subplots(1, 2, figsize = (10, 5), constrained_layout=True)
    legend=True
    with torch.no_grad():
        for x, y, c in test_loader_in:
            x = x.to(device)
            w = csvae.posteriorW(x).sample().cpu()
            if legend:
                for i in range(2):
                        label = labels[i]
                        color = colors[i] 
                        plot_scatter(axes[0],  w[y == i], [i]*sum(y==i), l=label, c=['tab:orange', 'tab:blue'])
                        plot_scatter(axes[1], w[c == i], [i]*sum(c==i), l=color, c=['r', 'g'])
                legend=False
            else:
                plot_scatter(axes[0], w, y)
                plot_scatter(axes[1], w, c, c=['r', 'g'])
    for i in range(2):
        axes[i].set_xlabel('$w_0$')
        axes[i].set_ylabel('$w_1$')
        axes[i].legend()
    plt.savefig(f'outputs/figures/conditional_independence.png')
    plt.close(fig)