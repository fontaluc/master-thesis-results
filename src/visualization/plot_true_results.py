from matplotlib import pyplot as plt
from plotting import plot_scatter
from torch.utils.data import DataLoader, TensorDataset
import torch
from model import DSVAE_prior_MNIST
import numpy as np
import yaml
import argparse
from src.utils import set_seed, seed_worker, counterfactuals

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-step', type=float, default=2)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()

    x_dim = 392 
    data = 'cmnist'
    color = True
    data_path = './data'

    plt.rcParams.update({'font.size': 22})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    for e in [0.05, 0.2, 0.5]:

        in_data = str(int(100*e))
            
        dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
        dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
        test_loader_in  = DataLoader(dset_test_in, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        
        fig, axes = plt.subplots(5, 4, figsize = (25, 30), constrained_layout=True)
        f, axarr = plt.subplots(10, 6, figsize=(6, 10), constrained_layout=True)
        
        # Plot 10 samples from the i.i.d. test set
        x_in, y_in, _ = next(iter(test_loader_in))
        labels_in = [4 if x.item() == 0 else 9 for x in y_in] 
        for i, ax in enumerate(axarr[:, 0]):
            ax.imshow(torch.cat((x_in[i].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(labels_in[i], fontsize=11)
        axarr[0, 0].set_title('Input', fontsize=11)
        
        i = 0
        for model in [('mmd', True), ('mmd', False), ('adversarial', True), ('adversarial', False), ('baseline', False), ]:
            
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

            axes[i, 0].set_title(title)                

            csvae = DSVAE_prior_MNIST(x_dim)
            csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
            csvae.load_state_dict(csvae_state)
            csvae = csvae.to(device)

            with torch.no_grad():
                for x, y, c in test_loader_in:
                    x = x.to(device)
                    w = csvae.posteriorW(x).sample().cpu()
                    z = csvae.posteriorZ(x).sample().cpu()
                    plot_scatter(axes[i, 0], w, y)
                    plot_scatter(axes[i, 1], w, c, c=['r', 'g'])
                    plot_scatter(axes[i, 2], z, y)
                    plot_scatter(axes[i, 3], z, c, c=['r', 'g'])
                            
            for j in range(2):
                axes[i, j].set_xlabel('$w_0$')
                axes[i, j].set_ylabel('$w_1$')
                axes[i, j+2].set_xlabel('$z_0$')
                axes[i, j+2].set_ylabel('$z_1$')

            # Counterfactuals
            x_in_CF = counterfactuals(csvae, device, x_in)
            for k, ax in enumerate(axarr[:, i+1]):
                ax.imshow(torch.cat((x_in_CF[k].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
                ax.axis('off')
            axarr[0, i+1].set_title(title.replace(', ', '\n'), fontsize=11)

            i += 1
        
        fig.suptitle(f'$p_e = {e}$')
        fig.savefig(f'outputs/exp1_latent_{e}.png', bbox_inches='tight')
        plt.close(fig)

        f.savefig(f'outputs/figures/exp1_cf_{e}.png')
        plt.close(f)