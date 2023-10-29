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
from sklearn.calibration import calibration_curve

def classification_measures(y_true, y_pred):
    TPR = sum((y_true == 1) & (y_pred == 1))/sum(y_true == 1)
    FPR = sum((y_true == 0) & (y_pred == 1))/sum(y_true == 0)
    PPV = sum((y_true == 1) & (y_pred == 1))/sum(y_pred == 1)
    NPV = sum((y_true == 0) & (y_pred ==0))/sum(y_pred == 0)
    return TPR, FPR, PPV, NPV

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)

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
    dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
    dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
    batch_size = hparams['batch_size']
    test_loader_in  = DataLoader(dset_test_in, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    torch.manual_seed(hparams["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    csvae = DSVAE_prior_MNIST(x_dim)
    csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
    csvae.load_state_dict(csvae_state)
    csvae = csvae.to(device)

    # Predictions on test set
    y_true = []
    y_pred = []
    y_probs = []
    c_true = []
    for x, y, c in test_loader_in:
        y_true += y.tolist()
        y_pred += csvae.classifier(x).flatten().tolist()
        c_true += c.tolist()
        with torch.no_grad():
            qw = csvae.posteriorW(x)
            w = qw.sample()
            qy = csvae.posteriorY(w)
            y_probs += qy.probs.flatten().tolist()
        
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_probs = np.array(y_probs, dtype=float)
    c_true = np.array(c_true, dtype=int)
    # TPR0, FPR0, PPV0, NPV0 = classification_measures(y_true[c_true == 0], y_pred[c_true == 0])
    # TPR1, FPR1, PPV1, NPV1 = classification_measures(y_true[c_true == 1], y_pred[c_true == 1])
    # dTPR = TPR0 - TPR1
    # dFPR = FPR0 - FPR1
    # dPPV = PPV0 - PPV1
    # dNPV = NPV0 - NPV1

    # f = open(f'{input_path}/fairness.txt', 'a')
    # f.write('SEPARATION \n')
    # f.write(f'Difference in TPR: {dTPR} \n')
    # f.write(f'Difference in FPR: {dFPR} \n')
    # f.write(f'Mean of absolute differences: {(abs(dTPR) + abs(dFPR))/2} \n')
    # f.write('\nSUFFICIENCY \n') 
    # f.write(f'Difference in PPV: {dPPV} \n')
    # f.write(f'Difference in NPV: {dNPV} \n')
    # f.write(f'Mean of absolute differences: {(abs(dPPV) + abs(dNPV))/2} \n')

    lr_x0, lr_y0 = calibration_curve(y_true[c_true == 0], y_probs[c_true == 0], n_bins=10)
    lr_x1, lr_y1 = calibration_curve(y_true[c_true == 1], y_probs[c_true == 1], n_bins=10)
    fig = plt.figure()
    plt.plot(lr_x0, lr_y0, marker='o', label='$c = 0$')
    plt.plot(lr_x1, lr_y1, marker='o', label='$c = 1$')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.savefig(f'{input_path}/calibration.png')
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize = (10, 5))
    for i in range(2):
        axs[i].hist(y_probs[c_true == i])
        axs[i].set_title(f'$c = {i}$')
        axs[i].set_xlabel('Mean predicted probability')
        axs[i].set_ylabel('Count')
    plt.savefig(f'{input_path}/histogram.png')
    plt.close()