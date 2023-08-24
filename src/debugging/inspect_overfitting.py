from src.plotting import visualize_latent_subspaces, visualize_decision_boundary
from src.models.model import DSVAE_prior_MNIST
from torch.utils.data import DataLoader, TensorDataset
import torch 
import argparse
import numpy as np
import yaml
from utils import get_cmnist_accuracy

def predict_bern(X, decoderY):
    X = torch.from_numpy(X).float()
    probs = decoderY(X)
    return probs.data.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-latent', action='store_true')
    parser.add_argument('-accuracy', action='store_true')
    parser.add_argument('-boundary', action='store_true')

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
    n = hparams['n']
    if n != 0:
     in_data += f'_{int(n*100)}'
     out_data += f'_{int(n*100)}'
    x_dim = 392
    data_path = './data'
    dataset_train = torch.load(f'{data_path}/{data}_train_{in_data}.pt')
    dataset_val = torch.load(f'{data_path}/{data}_valid_{in_data}.pt')
    dataset_test_in = torch.load(f'{data_path}/{data}_test_{in_data}.pt')
    dataset_test_out = torch.load(f'{data_path}/{data}_test_{out_data}.pt')
    dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
    dset_val = TensorDataset(dataset_val['images'], dataset_val['labels'], dataset_val['colors'])
    dset_test_in = TensorDataset(dataset_test_in['images'], dataset_test_in['labels'], dataset_test_in['colors'])
    dset_test_out = TensorDataset(dataset_test_out['images'], dataset_test_out['labels'], dataset_test_out['colors'])
    batch_size = hparams['batch_size']
    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(dset_val, batch_size=batch_size, shuffle=True)
    test_loader_in  = DataLoader(dset_test_in, batch_size=batch_size, shuffle=True)
    test_loader_out  = DataLoader(dset_test_out, batch_size=batch_size, shuffle=True)

    torch.manual_seed(hparams["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    csvae = DSVAE_prior_MNIST(x_dim)
    csvae_state = torch.load(f'{input_path}/csvae.pt', map_location=device)
    csvae.load_state_dict(csvae_state)
    csvae = csvae.to(device)

    if args.latent:
        visualize_latent_subspaces(csvae, train_loader, device, f'{input_path}/latent-train.png')
        visualize_latent_subspaces(csvae, val_loader, device, f'{input_path}/latent-valid.png')
        visualize_latent_subspaces(csvae, test_loader_out, device, f'{input_path}/latent-test-out.png')

    if args.accuracy:
        f = open(f'{input_path}/accuracies.txt', 'w')
        f.write(f'Train accuracy: {get_cmnist_accuracy(train_loader, csvae.classifier, device)}\n')
        f.write(f'Validation accuracy: {get_cmnist_accuracy(val_loader, csvae.classifier, device)}\n')
        f.write(f'Test accuracy (in): {get_cmnist_accuracy(test_loader_in, csvae.classifier, device)}\n') 
        f.write(f'Test accuracy (out): {get_cmnist_accuracy(test_loader_out, csvae.classifier, device)}')
        f.close()

    if args.boundary:
        x_train = dataset_train['images']
        y_train = dataset_train['labels'].long()
        x_val = dataset_val['images']
        y_val = dataset_val['labels'].long()

        with torch.no_grad():
            outputs_train = csvae(x_train.to(device), y_train.to(device))
            outputs_val = csvae(x_val.to(device), y_val.to(device))

        w_train, z_train = [outputs_train[k] for k in ["w", "z"]]
        w_val, z_val = [outputs_val[k] for k in ["w", "z"]]

        attributeNames = ['$w_1$', '$w_2$']
        classNames = ['Digit 4', 'Digit 9']
        visualize_decision_boundary(lambda x: predict_bern(x, csvae.decoderY), [w_train, w_val], [y_train, y_val], attributeNames, classNames, f'{input_path}/decision_boundary.png')
