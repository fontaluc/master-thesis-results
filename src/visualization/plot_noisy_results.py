from matplotlib import pyplot as plt
from plotting import plot_scatter, plot_samples
from torch.utils.data import DataLoader, TensorDataset
import torch
from model import DSVAE_prior_MNIST
import numpy as np
import yaml
import argparse
from utils import seed_worker, set_seed

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

    e = 0.2
    n = 0.2

    i = 0
    plt.rcParams.update({'font.size': 22})
    fig, axes = plt.subplots(6, 5, figsize = (25, 30))
    for model in [('mmd', True), ('mmd', False), ('adversarial', True), ('adversarial', False), ('baseline', False)]:
        input_path = f'outputs/train_{model[0]}/++seed={args.seed},'
        title = model[0]
        if model[0] == 'baseline':
            input_path += '+bw=1,+bx=1,+by=10,+bz=1,'
        else:
            if model[0] == 'mmd':
                 input_path += '+bmw=20,+bmz=20,+bw=1,+bx=1,+by=10,+bz=1,'
            else:
                input_path += '+bc=1,+bhw=20,+bhz=20,+bw=1,+bx=1,+by=10,+byz=1,+bz=1,'
            if model[1]:
                input_path += f'+conditional=true,'
                title += ', conditional'
            else:
                input_path += f'+conditional=false,'
                title += ', marginal'
                
        input_path += f'+e={e},+n={n}'                  

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

                # Plot counterfactuals for the first batch of test images
                if first:
                    x_orig = x
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
            for x, y, c in test_loader_out:
                x = x.to(device)
                w = csvae.posteriorW(x).sample().cpu()
                z = csvae.posteriorZ(x).sample().cpu()
                plot_scatter(axes[0, i], w, y, marker='x')
                plot_scatter(axes[1, i], w, c, marker='x', c=['r', 'g'])
                plot_scatter(axes[2, i], z, y, marker='x')
                plot_scatter(axes[3, i], z, c, marker='x', c=['r', 'g'])
                        
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

    fig.tight_layout()

    plt.savefig(f'outputs/figures/results_e={e}_n={n}.png', bbox_inches='tight')