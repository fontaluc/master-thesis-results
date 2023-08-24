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

def train(train_loader,csvae, aux_y, aux_c, optimizer_csvae, optimizer_aux_y, optimizer_aux_c, vi, device, epoch):
    training_epoch_data = defaultdict(list)
    csvae.train()
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y, c in train_loader:
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        csvae_loss, aux_y_loss, aux_c_loss, diagnostics, outputs = vi(csvae, aux_y, aux_c, x, y, c)
        
        optimizer_csvae.zero_grad()
        csvae_loss.backward()
        # Debug NaN gradients with Beta distribution
        # for param in csvae.decoder.parameters():
        #     print(param.grad)
        # torch.nn.utils.clip_grad_norm_(csvae.parameters(), max_norm=2.0)
        optimizer_csvae.step()

        optimizer_aux_y.zero_grad()
        aux_y_loss.backward()
        optimizer_aux_y.step()

        optimizer_aux_c.zero_grad()
        aux_c_loss.backward()
        optimizer_aux_c.step()
        
        # gather data for the current batch
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
            
    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        data = np.mean(training_epoch_data[k])
        wandb.log({f'{k}_train': data, 'epoch': epoch})
    
    return csvae_loss, aux_y_loss, aux_c_loss

def eval(valid_loader, csvae, aux_y, aux_c, vi, device, epoch):
    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        csvae.eval()
        
        # Just load a single batch from the test loader
        x, y, c = next(iter(valid_loader))
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        csvae_loss, aux_y_loss, aux_c_loss, diagnostics, outputs = vi(csvae, aux_y, aux_c, x, y, c)
        
        # gather data for the validation step
        for k, v in diagnostics.items():
            data = v.mean().item()
            wandb.log({f'{k}_valid': data, 'epoch': epoch})
    
    return csvae_loss, aux_y_loss, aux_c_loss, x, y, c, outputs

@hydra.main(
    version_base=None, config_path="../config", config_name="default_config.yaml"
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
    n = hparams['n']
    data = str(int(e*100)) 
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
    z_dim = 2
    aux_y = AUX(z_dim)
    aux_c = AUX(z_dim)

    # Evaluator: Variational Inference
    bx = hparams['bx']
    bw = hparams['bw']
    bz = hparams['bz']
    byw = hparams['by']
    bhw = hparams['bhw']
    bhz = hparams['bhz']
    byz = hparams['byz']
    bc = hparams['bc']
    conditional = hparams['conditional']
    vi = VI_DANN(bx, bw, bz, bhw, bhz, byw, byz, bc, conditional)

    # The Adam optimizer works really well with VAEs.
    lr = hparams['lr']
    optimizer_csvae = torch.optim.Adam(csvae.parameters(), lr=lr)
    optimizer_aux_y = torch.optim.Adam(aux_y.parameters(), lr=lr)
    optimizer_aux_c = torch.optim.Adam(aux_c.parameters(), lr=lr)

    epoch = 0
    num_epochs = hparams['epochs']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # move the model to the device
    csvae = csvae.to(device)
    aux_y = aux_y.to(device)
    aux_c = aux_c.to(device)
    vi.class_embeddings.to(device)
    vi.color_embeddings.to(device)

    # training..
    for epoch in range(num_epochs):
        _, _, _ = train(train_loader, csvae, aux_y, aux_c, optimizer_csvae, optimizer_aux_y, optimizer_aux_c, vi, device, epoch)
        csvae_valid_loss, aux_y_valid_loss, aux_c_valid_loss, x, y, c, outputs = eval(val_loader, csvae, aux_y, aux_c, vi, device, epoch)
        
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