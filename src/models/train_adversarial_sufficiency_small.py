from collections import defaultdict
from src.models.model import *
import hydra
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.plotting import log_cmnist_plots
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from src.utils import get_cmnist_accuracy, set_seed, seed_worker
import os
import shutil
import numpy as np

torch.autograd.set_detect_anomaly(True)

def train(train_loader, csvae, aux_w, aux_wzc, aux_z, aux_wzy, opt_csvae, opt_aux_w, opt_aux_wzc, opt_aux_z, opt_aux_wzy, vi, device, epoch):
    training_epoch_data = defaultdict(list)
    csvae.train()
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y, c in train_loader:
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        csvae_loss, aux_w_loss, aux_wzc_loss, aux_z_loss, aux_wzy_loss, diagnostics, outputs = vi(csvae, aux_w, aux_wzc, aux_z, aux_wzy, x, y, c)
        
        opt_csvae.zero_grad()
        csvae_loss.backward()
        opt_csvae.step()

        opt_aux_w.zero_grad()
        aux_w_loss.backward()
        opt_aux_w.step()

        opt_aux_wzc.zero_grad()
        aux_wzc_loss.backward()
        opt_aux_wzc.step()

        opt_aux_z.zero_grad()
        aux_z_loss.backward()
        opt_aux_z.step()

        opt_aux_wzy.zero_grad()
        aux_wzy_loss.backward()
        opt_aux_wzy.step()
        
        # gather data for the current batch
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
            
    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        data = np.mean(training_epoch_data[k])
        wandb.log({f'{k}_train': data, 'epoch': epoch})


def eval(valid_loader, csvae, aux_w, aux_wzc, aux_z, aux_wzy, vi, device, epoch):
    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        csvae.eval()
        
        # Just load a single batch from the test loader
        x, y, c = next(iter(valid_loader))
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        csvae_loss, aux_w_loss, aux_wzc_loss, aux_z_loss, aux_wzy_loss, diagnostics, outputs = vi(csvae, aux_w, aux_wzc, aux_z, aux_wzy, x, y, c)
        
        # gather data for the validation step
        for k, v in diagnostics.items():
            data = v.mean().item()
            wandb.log({f'{k}_valid': data, 'epoch': epoch})
    
    return x, y, c, outputs

@hydra.main(
    version_base=None, config_path="../../config", config_name="default_config.yaml"
)
def main(cfg):
    
    wandb.init(project="thesis")
    location = wandb.run.dir

    hparams = cfg

    set_seed(hparams["seed"])
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(hparams["seed"])

    e = hparams['e']
    data = str(int(e*100)) 
    if 'n' in hparams.keys():
        n = hparams['n']
        if n != 0:
            data += f'_{int(n*100)}'
    x_dim = 392
    data_path = f'{get_original_cwd()}/data'
    dataset_train = torch.load(f'{data_path}/cmnist_train_{data}.pt')
    dataset_val = torch.load(f'{data_path}/cmnist_valid_{data}.pt')
    dset_train = TensorDataset(dataset_train['images'], dataset_train['labels'], dataset_train['colors'])
    dset_val = TensorDataset(dataset_val['images'], dataset_val['labels'], dataset_val['colors'])
    batch_size = hparams['batch_size']
    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader  = DataLoader(dset_val, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

    m0 = hparams['m0']
    s0 = hparams['s0']
    m1 = hparams['m1']
    s1 = hparams['s1']
    conv = hparams['conv']
    csvae = DSCVAE_prior_MNIST(m0=m0, s0=s0, m1=m1, s1=s1) if conv else DSVAE_prior_MNIST(x_dim=x_dim, m0=m0, s0=s0, m1=m1, s1=s1)
    w_dim = 2
    z_dim = 2
    lin = hparams['linear']
    aux_w = AUX(w_dim, linear=lin)
    aux_wzc = AUX(w_dim + z_dim + 1, linear=lin)
    aux_z = AUX(z_dim, linear=False)
    aux_wzy = AUX(w_dim + z_dim + 1, linear=False)

    # Evaluator: Variational Inference
    bx = hparams['bx']
    bw = hparams['bw']
    bz = hparams['bz']
    by = hparams['by']
    bhw = hparams['bhw']
    bhz = hparams['bhz']
    vi = VI_sufficiency(bx, bw, bz, bhw, bhz, by) 

    # The Adam optimizer works really well with VAEs.
    lr = hparams['lr']
    opt_csvae = torch.optim.Adam(csvae.parameters(), lr=lr)
    opt_aux_w = torch.optim.Adam(aux_w.parameters(), lr=lr)
    opt_aux_wzc = torch.optim.Adam(aux_wzc.parameters(), lr=lr)
    opt_aux_z = torch.optim.Adam(aux_z.parameters(), lr=lr)
    opt_aux_wzy = torch.optim.Adam(aux_wzy.parameters(), lr=lr)

    epoch = 0
    num_epochs = hparams['epochs']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # move the model to the device
    csvae = csvae.to(device)
    aux_w = aux_w.to(device)
    aux_wzc = aux_wzc.to(device)
    aux_z = aux_z.to(device)
    aux_wzy = aux_wzy.to(device)

    # training..
    for epoch in range(num_epochs):
        train(train_loader, csvae, aux_w, aux_wzc, aux_z, aux_wzy, opt_csvae, opt_aux_w, opt_aux_wzc, opt_aux_z, opt_aux_wzy, vi, device, epoch)
        x, y, c, outputs = eval(val_loader, csvae, aux_w, aux_wzc, aux_z, aux_wzy, vi, device, epoch)
        
        train_acc = get_cmnist_accuracy(train_loader, csvae.classifier, device)
        wandb.log({'train_acc': train_acc, 'epoch': epoch})         
        log_cmnist_plots(x, y, c, outputs, m0, s0, m1, s1, epoch)

    torch.save(csvae.state_dict(), 'csvae.pt')
    wandb.finish()
    # Remove local media directory
    path = os.path.join(location, 'media')
    shutil.rmtree(path)
    

if __name__ == "__main__":   
    main()