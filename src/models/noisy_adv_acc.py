from matplotlib import pyplot as plt
from src.plotting import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from model import DSVAE_prior_MNIST, MMD, Adversarial, AUX
import argparse
import numpy as np
import yaml
from src.utils import median_heuristic, get_cmnist_accuracy
from typing import *
from sklearn.metrics import accuracy_score

def get_adversarial_error(csvae, lr, device, train_loader, valid_loader, test_loader, path, num_epochs = 10, validation_every_steps = 500):
    # aux_y = Adversarial(z_dim=3)
    # aux_c = Adversarial(z_dim=3)
    aux_y = AUX(z_dim=2)
    aux_c = AUX(z_dim=2)
    aux_y = aux_y.to(device)
    aux_c = aux_c.to(device)
    optimizer_y = torch.optim.Adam(aux_y.parameters(), lr=lr)
    optimizer_c = torch.optim.Adam(aux_c.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_valid = len(valid_loader.dataset)
    n_test = len(test_loader.dataset)
    
    step = 0
    aux_y.train()
    aux_c.train()

    train_accuracies_y = []
    valid_accuracies_y = []
    train_accuracies_c = []
    valid_accuracies_c = []
            
    for epoch in range(num_epochs):
        
        train_accuracies_batches_y = []
        train_accuracies_batches_c = []
        
        for x, y, c in train_loader:

            x = x.to(device)
            y = y.to(device)
            c = c.to(device)

            qw = csvae.posteriorW(x)
            w = qw.rsample()
            qz = csvae.posteriorZ(x)
            z = qz.rsample()
            
            # output_y = aux_y(zc)
            # loss_y = loss_fn(output_y, y.long())
            qy = aux_y(z)
            log_qy = qy.log_prob(y)
            aux_y_loss = log_qy.mean()
            optimizer_y.zero_grad()
            aux_y_loss.backward()
            optimizer_y.step()

            # output_c = aux_y(wy)
            # loss_c = loss_fn(output_c, c.long())
            qc = aux_c(w)
            log_qc = qc.log_prob(c)
            aux_c_loss = log_qc.mean()
            optimizer_c.zero_grad()
            aux_c_loss.backward()
            optimizer_c.step()
            
            step += 1
            
            # Compute accuracy.
            # predictions_y = output_y.max(1)[1]
            # train_accuracies_batches_y.append(accuracy_score(y.detach().cpu().numpy(), predictions_y.cpu().numpy()))

            # predictions_c = output_c.max(1)[1]
            # train_accuracies_batches_c.append(accuracy_score(c.cpu().numpy(), predictions_c.cpu().numpy()))

            train_accuracies_batches_y.append(torch.exp(log_qy).mean().item())
            train_accuracies_batches_c.append(torch.exp(log_qc).mean().item())
            
            if step % validation_every_steps == 0:
                
                # Append average training accuracy to list.
                train_accuracies_y.append(np.mean(train_accuracies_batches_y))
                train_accuracies_c.append(np.mean(train_accuracies_batches_c))
                
                train_accuracies_batches_y = []
                train_accuracies_batches_c = []
            
                # Compute accuracies on validation set.
                valid_accuracies_batches_y = []
                valid_accuracies_batches_c = []
                with torch.no_grad():
                    aux_y.eval()
                    aux_c.eval()
                    for x, y, c in valid_loader:

                        x = x.to(device)
                        y = y.to(device)
                        c = c.to(device)

                        qw = csvae.posteriorW(x)
                        w = qw.rsample()
                        qz = csvae.posteriorZ(x)
                        z = qz.rsample()

                        # output_y = aux_y(zc)
                        # loss_y = loss_fn(output_y, y.long())
                        # predictions_y = output_y.max(1)[1]

                        # output_c = aux_c(wy)
                        # loss_c = loss_fn(output_c, c.long())
                        # predictions_c = output_c.max(1)[1]

                        # # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        # valid_accuracies_batches_y.append(accuracy_score(y.detach().cpu().numpy(), predictions_y.detach().cpu().numpy()) * len(y))
                        # valid_accuracies_batches_c.append(accuracy_score(c.detach().cpu().numpy(), predictions_c.detach().cpu().numpy()) * len(c))

                        qy = aux_y(z)
                        log_qy = qy.log_prob(y)

                        qc = aux_c(w)
                        log_qc = qc.log_prob(c)

                        valid_accuracies_batches_y.append(torch.exp(log_qy).mean().item()*len(y))
                        valid_accuracies_batches_c.append(torch.exp(log_qc).mean().item()*len(c))

                    aux_y.train()
                    aux_c.train()
                    
                # Append average validation accuracy to list.
                valid_accuracies_y.append(np.sum(valid_accuracies_batches_y) / n_valid)
                valid_accuracies_c.append(np.sum(valid_accuracies_batches_c) / n_valid)             
        
                print(f"Step {step:<5}   training accuracy: {train_accuracies_y[-1]:.3f}, {train_accuracies_c[-1]:.3f}")
                print(f"             validation accuracy: {valid_accuracies_y[-1]:.3f}, {valid_accuracies_c[-1]:.3f}")

    print("Finished training.")

    steps = (np.arange(len(train_accuracies_y), dtype=int) + 1) * validation_every_steps

    plt.figure()
    plt.plot(steps, train_accuracies_y, label='train')
    plt.plot(steps, valid_accuracies_y, label='validation')
    plt.xlabel('Training steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Train and validation accuracy for predicting y")
    plt.savefig(f'{path}/accuracies_y.png')

    plt.figure()
    plt.plot(steps, train_accuracies_c, label='train')
    plt.plot(steps, valid_accuracies_c, label='validation')
    plt.xlabel('Training steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Train and validation accuracy for predicting c")
    plt.savefig(f'{path}/accuracies_c.png')

    # Evaluate test set
    with torch.no_grad():
        aux_y.eval()
        aux_c.eval()
        test_accuracies_y = []
        test_accuracies_c = []
        for x, y, c in test_loader:

            x = x.to(device)
            y = y.to(device)
            c = c.to(device)

            qw = csvae.posteriorW(x)
            w = qw.rsample()
            qz = csvae.posteriorZ(x)
            z = qz.rsample()
            
            # output_y = aux_y(zc)
            # loss_y = loss_fn(output_y, y.long())
            # predictions_y = output_y.max(1)[1]

            # output_c = aux_c(wy)
            # loss_c = loss_fn(output_c, c.long())
            # predictions_c = output_c.max(1)[1]

            # # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=True).
            # test_accuracies_y.append(accuracy_score(y.cpu().numpy(), predictions_y.cpu().numpy()) * len(y))
            # test_accuracies_c.append(accuracy_score(c.cpu().numpy(), predictions_c.cpu().numpy()) * len(c))

            qy = aux_y(z)
            log_qy = qy.log_prob(y)

            qc = aux_c(w)
            log_qc = qc.log_prob(c)

            test_accuracies_y.append(torch.exp(log_qy).mean().item()*len(y))
            test_accuracies_c.append(torch.exp(log_qc).mean().item()*len(c))

        test_accuracy_y = np.sum(test_accuracies_y) / n_test
        test_accuracy_c = np.sum(test_accuracies_c) / n_test
        return test_accuracy_y, test_accuracy_c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-step', type=float, default=2)

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
    e_out = 1 - e_in
    in_data = str(int(100*e_in))
    out_data = str(int(100*e_out))
    bx = hparams['bx']

    n = hparams['n']
    if n != 0:
     in_data += f'_{int(n*100)}'

    x_dim = 392
    data_path = './data'
    dataset_train = torch.load(f'{data_path}/{data}_train_{in_data}.pt')
    dataset_val = torch.load(f'{data_path}/{data}_valid_{in_data}.pt')
    dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
    dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
    dset_val = TensorDataset(dataset_val['images'], dataset_val['labels'], dataset_val['colors'])
    dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
    batch_size = hparams['batch_size']
    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(dset_val, batch_size=batch_size, shuffle=True)
    test_loader_in  = DataLoader(dset_test_in, batch_size=batch_size, shuffle=True)

    torch.manual_seed(hparams["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    csvae = DSVAE_prior_MNIST(x_dim)
    csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
    csvae.load_state_dict(csvae_state)
    csvae = csvae.to(device)

    lr = hparams['lr']

    acc_y, acc_c = get_adversarial_error(csvae, lr, device, train_loader, val_loader, test_loader_in, input_path)
    print(acc_y, acc_c)