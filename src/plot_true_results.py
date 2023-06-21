from matplotlib import pyplot as plt
from plotting import plot_scatter, plot_samples
from torch.utils.data import DataLoader, TensorDataset
import torch
from model import DSVAE_prior_MNIST
import numpy as np
import yaml
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-step', type=float, default=2)
    args = parser.parse_args()

    x_dim = 392 
    data = 'cmnist'
    color = True
    data_path = './data'

    for e in [0.05, 0.2, 0.5]:
        i = 0
        fig, axes = plt.subplots(6, 5, figsize = (25, 30))
        for model in [('baseline', False), ('mmd', False), ('mmd', True), ('adversarial', False), ('adversarial', True)]:
            input_path = f'outputs/train_{model[0]}/'
            title = model[0]
            if model[0] != 'baseline':
                if model[1]:
                    input_path += f'+conditional=true,'
                    title += ', conditional'
                else:
                    input_path += f'+conditional=false,'
                    title += ', marginal'
                    
            input_path += f'+e={e}'                  

            with open(f"{input_path}/.hydra/config.yaml", "r") as f:
                cfg = yaml.safe_load(f)

            with open(f"{input_path}/.hydra/hydra.yaml", "r") as f:
                cfg_hydra = yaml.safe_load(f)

            hparams = cfg
            
            in_data = str(int(100*e))
            
            dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
            dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
            batch_size = hparams['batch_size']
            test_loader_in  = DataLoader(dset_test_in, batch_size=batch_size, shuffle=True)

            torch.manual_seed(hparams["seed"])

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            csvae = DSVAE_prior_MNIST(x_dim)
            csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
            csvae.load_state_dict(csvae_state)
            csvae = csvae.to(device) 

            first = True        

            with torch.no_grad():
                for x, y, c in test_loader_in:
                    x = x.to(device)
                    w = csvae.posteriorW(x).sample().cpu()
                    z = csvae.posteriorZ(x).sample().cpu()
                    plot_scatter(axes[0, i], w, y)
                    plot_scatter(axes[1, i], w, c, c=['r', 'g'])
                    plot_scatter(axes[2, i], z, y)
                    plot_scatter(axes[3, i], z, c, c=['r', 'g'])

                    if first:
                        layer = csvae.decoderY[0]
                        a = layer.weight
                        b = layer.bias
                        alpha = -(w @ a.view(2, 1) + b)/(a @ a.view(2, 1))
                        dw = args.step*alpha*a 
                        w_CF = w + dw
                        x_CF = csvae.decode(w_CF, z).cpu()
                        plot_samples(axes[4, i], x_CF, color)
                        if i == 0:
                            plot_samples(axes[5, 2], x, color)
                            axes[5, 2].set_title('Original')
                    first = False
                            
            for j in range(2):
                axes[j, i].set_xlabel('$w_0$')
                axes[j, i].set_ylabel('$w_1$')
                axes[j+2, i].set_xlabel('$z_0$')
                axes[j+2, i].set_ylabel('$z_1$')

            axes[0, i].set_title(title)

            i += 1  

        idx = [0, 1, 3, 4]
        for i in idx:
            fig.delaxes(axes[5, i])
        plt.savefig(f'outputs/results_e={e}.png', bbox_inches='tight')