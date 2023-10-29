from collections import defaultdict
from src.models.model import *
import hydra
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from src.plotting import log_cmnist_plots
from hydra.utils import get_original_cwd
from src.utils import get_cmnist_accuracy, median_heuristic, set_seed, seed_worker
import numpy as np
import os
import shutil

torch.autograd.set_detect_anomaly(True)

def train(train_loader, csvae, aux_y, aux_zy, opt_csvae, opt_aux_y, opt_aux_zy, vi, device, epoch):
    training_epoch_data = defaultdict(list)
    csvae.train()
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y, c in train_loader:
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        loss, aux_y_loss, aux_zy_loss, diagnostics, outputs = vi(csvae, aux_y, aux_zy, x, y, c)
        
        opt_csvae.zero_grad()
        loss.backward()
        opt_csvae.step()

        opt_aux_y.zero_grad()
        aux_y_loss.backward()
        opt_aux_y.step()

        opt_aux_zy.zero_grad()
        aux_zy_loss.backward()
        opt_aux_zy.step()
        
        # gather data for the current batch
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
            
    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        data = np.mean(training_epoch_data[k])
        wandb.log({f'{k}_train': data, 'epoch': epoch})

def eval(valid_loader, csvae, aux_y, aux_zy, vi, device, epoch):
    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        csvae.eval()
        
        # Just load a single batch from the test loader
        x, y, c = next(iter(valid_loader))
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        csvae_loss, aux_y_loss, aux_zy_loss, diagnostics, outputs = vi(csvae, aux_y, aux_zy, x, y, c)
        
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
    csvae = DSVAE_prior_MNIST(x_dim=x_dim, m0=m0, s0=s0, m1=m1, s1=s1)
    aux_y = AUX_W(y_dim=1)
    aux_zy = AUX_W(y_dim=3)

    # Evaluator: Variational Inference
    bx = hparams['bx']
    bw = hparams['bw']
    bz = hparams['bz']
    by = hparams['by']
    bmz = hparams['bmz']
    bmw = hparams['bmw']
    bh = hparams['bh']
    n_median = hparams['n_median']
    conditional = True
    vi = VI_MMD_cond_aux(bx, bw, bz, by, bmw, bmz, bh)

    # The Adam optimizer works really well with VAEs.
    lr = hparams['lr']
    opt_csvae = torch.optim.Adam(csvae.parameters(), lr=lr)
    opt_aux_y = torch.optim.Adam(aux_y.parameters(), lr=lr)
    opt_aux_zy = torch.optim.Adam(aux_zy.parameters(), lr=lr)

    num_epochs = hparams['epochs']

    if hparams['scheduler']:
        num_epochs = 300
        optim.lr_scheduler.MultiStepLR(optimizer, milestones=[pow(3, i) for i in range(5)], gamma=pow(0.1, 1/7))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # move the model to the device
    csvae = csvae.to(device)
    aux_y = aux_y.to(device)
    aux_zy = aux_zy.to(device)

    # training..
    for epoch in range(num_epochs):

        if epoch % n_median == 0:
            lengthscales = median_heuristic(dataset_train, csvae, device, conditional)
            vi.set_lengthscale_(*lengthscales)
            if conditional:
                lw0, lw1, lz0, lz1 = lengthscales
                wandb.log({'lw0': lw0, 'epoch': epoch})
                wandb.log({'lz0': lz0, 'epoch': epoch})
                wandb.log({'lw1': lw1, 'epoch': epoch})
                wandb.log({'lz1': lz1, 'epoch': epoch}) 
            else:
                lw, lz = lengthscales
                wandb.log({'lw': lw, 'epoch': epoch})
                wandb.log({'lz': lz, 'epoch': epoch})  

        train(train_loader, csvae, aux_y, aux_zy, opt_csvae, opt_aux_y, opt_aux_zy, vi, device, epoch)
        x, y, c, outputs = eval(val_loader, csvae, aux_y, aux_zy, vi, device, epoch)
        
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