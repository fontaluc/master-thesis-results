from matplotlib import pyplot as plt
from plotting import plot_scatter
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

    x_dim = 392 
    data = 'cmnist'
    color = True
    data_path = './data'

    e = 0.2
    n = 0.2

    in_data = str(int(100*e))
    out_data = str(int(100*(1-e)))
    
    dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
    dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
    test_loader_in  = DataLoader(dset_test_in, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    x_in, y_in, _ = next(iter(test_loader_in))
    labels_in = [4 if x.item() == 0 else 9 for x in y_in]

    dataset_test_out = torch.load(f'{data_path}/{data}_test_{out_data}.pt')
    dset_test_out = TensorDataset(dataset_test_out['images'], dataset_test_out['labels'], dataset_test_out['colors'])
    test_loader_out  = DataLoader(dset_test_out, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels = ['4', '9']
    colors = ['red', 'green']
    plt.rcParams.update({'font.size': 15})

    bw=2
    by=1

    fig1, axes1 = plt.subplots(5, 1, figsize=(5, 15), constrained_layout=True)
    fig2, axes2 = plt.subplots(5, 1, figsize=(5, 15), constrained_layout=True)
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

    i=0
    for model in [('mmd', True), ('mmd', False), ('adversarial', True), ('adversarial', False), ('baseline', False)]:

        bm = 20 if model[1] else 1

        input_path = f'outputs/train_{model[0]}/++seed={args.seed},'
        title = model[0]
        if model[0] == 'baseline':
            input_path += f'+bw={bw},+bx=1,+by={by},+bz=1,'
        else:
            if model[0] == 'mmd':
                 input_path += f'+bmw={bm},+bmz={bm},+bw={bw},+bx=1,+by={by},+bz=1,'
            else:
                input_path += f'+bc=1,+bhw={bm},+bhz={bm},+bw={bw},+bx=1,+by={by},+byz=1,+bz=1,'
            if model[1]:
                input_path += f'+conditional=true,'
                title += ', conditional'
            else:
                input_path += f'+conditional=false,'
                title += ', marginal'
                
        input_path += f'+e={e},+n={n}'                  

        csvae = DSVAE_prior_MNIST(x_dim)
        csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
        csvae.load_state_dict(csvae_state)
        csvae = csvae.to(device) 

        with torch.no_grad():
            for x, y, c in test_loader_out:
                x = x.to(device)
                w = csvae.posteriorW(x).sample().cpu()
                z = csvae.posteriorZ(x).sample().cpu()
                plot_scatter(axes1[i], w, y)
            for x, y, c in test_loader_in:
                x = x.to(device)
                w = csvae.posteriorW(x).sample().cpu()
                z = csvae.posteriorZ(x).sample().cpu()
                plot_scatter(axes2[i], z, c, c=['r', 'g'])

        axes1[i].set_xlabel('$w_0$')
        axes1[i].set_ylabel('$w_1$')
        axes1[i].set_title(title)
        axes2[i].set_xlabel('$z_0$')
        axes2[i].set_ylabel('$z_1$')
        axes2[i].set_title(title)

        x_in_CF = counterfactuals(csvae, device, x_in)
        for k, ax in enumerate(axarr[:, i+1]):
            ax.imshow(torch.cat((x_in_CF[k].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0))
            ax.axis('off')
        axarr[0, i+1].set_title(title.replace(', ', '\n'), fontsize=11)

        i += 1

    fig1.savefig('outputs/exp2_latent_ood.png')
    plt.close(fig1)
    fig2.savefig('outputs/exp2_latent_iid.png')
    plt.close(fig2)
    f.savefig('outputs/exp2_cf.png')
    plt.close(f)
        