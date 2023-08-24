from matplotlib import pyplot as plt
from plotting import plot_scatter, plot_sample_column
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
    
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    x_dim = 392 
    data = 'cmnist'
    color = True
    data_path = './data'

    labels = ['4', '9']
    colors = ['red', 'green']
    plt.rcParams.update({'font.size': 22})
    for model in [('mmd', True), ('mmd', False), ('adversarial', True), ('adversarial', False), ('baseline', False)]:
        for e in [0.05, 0.2, 0.5]:
            fig, axes = plt.subplots(2, 2, figsize = (10, 10))

            input_path = f'outputs/train_{model[0]}/++seed={args.seed},'
            title = model[0]
            if model[0] == 'baseline':
                input_path += '+bw=10,+bx=1,+by=10,+bz=1,'
            else:
                if model[0] == 'mmd':
                    input_path += '+bmw=10,+bmz=10,+bw=10,+bx=1,+by=10,+bz=1,'
                else:
                    input_path += '+bc=1,+bhw=10,+bhz=10,+bw=10,+bx=1,+by=10,+byz=1,+bz=1,'
                if model[1]:
                    input_path += f'+conditional=true,'
                    title += ', conditional'
                else:
                    input_path += f'+conditional=false,'
                    title += ', marginal'
                    
            input_path += f'+e={e}' 
            title += f', e = {e}'                 

            with open(f"{input_path}/.hydra/config.yaml", "r") as f:
                cfg = yaml.safe_load(f)

            with open(f"{input_path}/.hydra/hydra.yaml", "r") as f:
                cfg_hydra = yaml.safe_load(f)

            hparams = cfg
            
            in_data = str(int(100*e))
            out_data = str(int(100*(1-e)))
            
            dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
            dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
            batch_size = hparams['batch_size']
            test_loader_in  = DataLoader(dset_test_in, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

            dataset_test_out = torch.load(f'{data_path}/{data}_test_{out_data}.pt')
            dset_test_out = TensorDataset(dataset_test_out['images'], dataset_test_out['labels'], dataset_test_out['colors'])
            test_loader_out  = DataLoader(dset_test_out, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            csvae = DSVAE_prior_MNIST(x_dim)
            csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
            csvae.load_state_dict(csvae_state)
            csvae = csvae.to(device) 

            # Latent subspaces
            fig, axes = plt.subplots(4, 2, figsize=(10, 20), constrained_layout=True)
            with torch.no_grad():
                for x, y, c in test_loader_in:
                    x = x.to(device)
                    w = csvae.posteriorW(x).sample().cpu()
                    z = csvae.posteriorZ(x).sample().cpu()
                    plot_scatter(axes[0, 0], w, y)
                    plot_scatter(axes[1, 0], w, c, c=['r', 'g'])
                    plot_scatter(axes[2, 0], z, y)
                    plot_scatter(axes[3, 0], z, c, c=['r', 'g'])
                for x, y, c in test_loader_out:
                    x = x.to(device)
                    w = csvae.posteriorW(x).sample().cpu()
                    z = csvae.posteriorZ(x).sample().cpu()
                    plot_scatter(axes[0, 1], w, y)
                    plot_scatter(axes[1, 1], w, c, c=['r', 'g'])
                    plot_scatter(axes[2, 1], z, y)
                    plot_scatter(axes[3, 1], z, c, c=['r', 'g'])
                            
            for j in range(2):
                for i in range(2):
                    axes[j, i].set_xlabel('$w_0$')
                    axes[j, i].set_ylabel('$w_1$')
                    axes[j+2, i].set_xlabel('$z_0$')
                    axes[j+2, i].set_ylabel('$z_1$')
            axes[0, 0].set_title('i.i.d')
            axes[0, 1].set_title('o.o.d')
            fig.suptitle(title)
            plt.savefig(f'outputs/exp1_{model[0]}_{model[1]}_{e}.png')
            plt.close(fig)

            # Counterfactuals 
            x_in, y_in, _ = next(iter(test_loader_in))
            labels_in = [4 if x.item() == 0 else 9 for x in y_in]
            x_in_CF = counterfactuals(csvae, device, x_in)  

            f, axarr = plt.subplots(15, 2, figsize=(2, 15), constrained_layout=True)
            for i, ax in enumerate(axarr[:, 0]):
                ax.imshow(torch.cat((x_in[i].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(labels_in[i], fontsize=11)
            for i, ax in enumerate(axarr[:, 1]):
                ax.imshow(torch.cat((x_in_CF[i].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
                ax.axis('off')
            axarr[0, 0].set_title('Input')
            axarr[0, 1].set_title('CF')
            f.suptitle('i.i.d.')
            plt.savefig(f'outputs/exp1_{model[0]}_{model[1]}_iid_{e}.png')
            plt.close(f)

            # x_out, y_out, _ = next(iter(test_loader_out))
            # x_out_CF = counterfactuals(csvae, device, x_out) 
            # f, axarr = plt.subplots(15, 2, figsize=(2, 20), constrained_layout=True)
            # for i, ax in enumerate(axarr[:, 0]):
            #     ax.imshow(torch.cat((x_out[i].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #     ax.set_ylabel(y_out[i])
            # for i, ax in enumerate(axarr[:, 1]):
            #     ax.imshow(torch.cat((x_out_CF[i].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
            #     ax.axis('off')
            # axarr[0, 0].set_title('Input')
            # axarr[0, 1].set_title('CF')
            # f.suptitle('o.o.d.')
            # plt.savefig(f'outputs/exp1_{model[0]}_{model[1]}_ood.png')
            # plt.close(f)
        