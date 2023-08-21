import os
from typing import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from torchvision.utils import make_grid
import wandb
import torch

## Useful functions

def plot_samples(ax, x, color):
    x = x.to('cpu')
    nrow = int(np.sqrt(x.size(0)))
    if not color:
        x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=nrow).permute(1, 2, 0)
    else:
        x_grid = make_grid(torch.cat((x.view(-1, 2, 14, 14), torch.zeros(x.shape[0], 1, 14, 14)), dim = 1), nrow=nrow).permute(1, 2, 0)
    ax.imshow(x_grid)
    ax.axis('off')

def plot_2d_latents(ax, qw, w, y, m0, s0, m1, s1):
    w = w.to('cpu')
    y = y.to('cpu')
    scale_factor_0 = 2*s0
    scale_factor_1 = 2*s1
    scale_factor = np.where(y == 1, scale_factor_0, scale_factor_1)
    batch_size = w.shape[0]
    palette = sns.color_palette()
    colors = [palette[int(l)] for l in y]

    # plot prior
    prior_0 = plt.Circle((m0, m0), scale_factor_0, color='gray', fill=True, alpha=0.1)
    ax.add_artist(prior_0)

    prior_1 = plt.Circle((m1, m1), scale_factor_1, color='gray', fill=True, alpha=0.1)
    ax.add_artist(prior_1)

    # plot data points
    mus, sigmas = qw.mu.to('cpu'), qw.sigma.to('cpu')
    mus = [mus[i].numpy().tolist() for i in range(batch_size)]
    sigmas = [sigmas[i].numpy().tolist() for i in range(batch_size)]

    posteriors = [
        plt.matplotlib.patches.Ellipse(mus[i], *(scale_factor[i] * s for s in sigmas[i]), color=colors[i], fill=False,
                                       alpha=0.3) for i in range(batch_size)]
    for p in posteriors:
        ax.add_artist(p)

    ax.scatter(w[:, 0], w[:, 1], color=colors)
    m_min = min(m0, m1)
    m_max = max(m0, m1)
    ax.set_xlim([m_min - 3, m_max + 3])
    ax.set_ylim([m_min - 3, m_max + 3])
    ax.set_aspect('equal', 'box')

def plot_latents(ax, z, y):
    z = z.to('cpu')
    y = y.to('cpu')
    palette = sns.color_palette()
    colors = [palette[int(l)] for l in y]
    z = TSNE(n_components=2).fit_transform(z)
    ax.scatter(z[:, 0], z[:, 1], color=colors)

def visualize_latent(ax, outputs, var, y, m0, s0, m1, s1):
    w = outputs[var]
    if w.shape[1] == 2:
        ax.set_title(r'Latent Samples $\mathbf{{{0}}} \sim q_\phi(\mathbf{{{0}}} | \mathbf{{x}})$'.format(var))
        qw = outputs['q' + var]
        plot_2d_latents(ax, qw, w, y, m0, s0, m1, s1)
    else:
        ax.set_title(r'Latent Samples $\mathbf{{{0}}} \sim q_\phi(\mathbf{{{0}}} | \mathbf{{x}})$ (t-SNE)'.format(var))
        plot_latents(ax, w, y)
    return ax

def log_reconstruction(x, outputs, epoch, tmp_img="tmp_reconstruction.png", figsize=(10, 5), color=False):
    fig = plt.figure(figsize = figsize)

    # plot the observation
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title(r'Observation $\mathbf{x}$')
    plot_samples(ax, x, color)

    # plot posterior samples
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title(
        r'Reconstruction $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{w}, \mathbf{z}), \mathbf{w} \sim q_\phi(\mathbf{w} | \mathbf{x}, \mathbf{y}), \mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')
    x = outputs['x']
    x_sample = x.to('cpu')
    plot_samples(ax, x_sample, color)

    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)
    wandb.log({'reconstruction': wandb.Image(tmp_img), 'epoch': epoch})
    os.remove(tmp_img)
    
def log_latent(y, outputs, m0, s0, m1, s1, epoch, tmp_img="tmp_latent.png", label = "y", figsize=(10, 5)):
    fig = plt.figure(figsize = figsize)

    # plot the latent samples
    try:
        ax = fig.add_subplot(1, 2, 1)
        visualize_latent(ax, outputs, 'w', y, m0, s0, m1, s1)

        ax = fig.add_subplot(1, 2, 2)
        visualize_latent(ax, outputs, 'z', y, 0, 1, 0, 1)
        
    except Exception as e:
        print(f"Could not generate the plot of the latent samples because of exception")
        print(e)

    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)
    wandb.log({f'latent_{label}': wandb.Image(tmp_img), 'epoch': epoch})
    os.remove(tmp_img)

def log_cmnist_plots(x, y, c, outputs, m0, s0, m1, s1, epoch, figsize = (10, 5)):
    log_reconstruction(x, outputs, epoch, figsize = figsize, color = True)
    log_latent(y, outputs, m0, s0, m1, s1, epoch, label = "y", figsize = figsize)
    log_latent(c, outputs, m0, s0, m1, s1, epoch, label = "c", figsize = figsize)

def plot_scatter(ax, X, y, marker='o', l=None, c=['tab:orange', 'tab:blue']):
    colors = [c[int(i)] for i in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, marker=marker, label=l, alpha=.33)

def plot_latent_subspaces(axes, csvae, dataloader, device, marker='o', legend=True, labels = ['4', '9'], colors = ['red', 'green']):   
    with torch.no_grad():
        for x, y, c in dataloader:
            x = x.to(device)
            w = csvae.posteriorW(x).sample().cpu()
            z = csvae.posteriorZ(x).sample().cpu()
            if legend:
                for i in range(2):
                        label = labels[i]
                        color = colors[i] 
                        plot_scatter(axes[0, 0],  w[y == i], [i]*sum(y==i), marker=marker, l=label, c=['tab:orange', 'tab:blue'])
                        plot_scatter(axes[0, 1], w[c == i], [i]*sum(c==i), marker=marker, l=color, c=['r', 'g'])
                legend=False
            else:
                plot_scatter(axes[0, 0], w, y, marker=marker)
                plot_scatter(axes[0, 1], w, c, marker=marker, c=['r', 'g'])
            plot_scatter(axes[1, 0], z, y, marker=marker)
            plot_scatter(axes[1, 1], z, c, marker=marker, c=['r', 'g'])
    
    for i in range(2):
        axes[0, i].set_xlabel('$w_0$')
        axes[0, i].set_ylabel('$w_1$')
        axes[1, i].set_xlabel('$z_0$')
        axes[1, i].set_ylabel('$z_1$')
    axes[0, 0].legend()
    axes[0, 1].legend()
    return axes

def visualize_latent_subspaces(csvae, dataloader, device, filename, labels = ['4', '9'], colors = ['red', 'green']):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    plot_latent_subspaces(axes, csvae, dataloader, device, labels=labels, colors=colors)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    
def visualize_label_counterfactuals(data_loader, csvae, device, path, figsize = (10, 5), color=False, step=None):
    """
    Visualize a batch of counterfactual images with respect to the digit label
    """
    x, y, c = next(iter(data_loader))
    y_CF = 1 - y

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].set_title('Observations')
    plot_samples(axes[0], x, color)

    with torch.no_grad():

        x = x.to(device)
        z = csvae.posteriorZ(x).sample()

        if step is None:
            # Sample w from the prior using the counterfactual label
            w_CF = csvae.priorW(y_CF).sample()
        
        else:
            layer = csvae.decoderY[0]
            a = layer.weight
            b = layer.bias
            w = csvae.posteriorW(x).sample()
            alpha = -(w @ a.view(2, 1) + b)/(a @ a.view(2, 1))
            dw = step*alpha*a 
            w_CF = w + dw

        x_CF = csvae.decode(w_CF, z).cpu()

    axes[1].set_title('Counterfactuals')
    plot_samples(axes[1], x_CF, color)

    plt.savefig(path)
    plt.close(fig)

def counterfactual_projection(train_loader, valid_loader, idx, csvae, step, path):
    """
    Show a single image from the validation set projected onto the decision boundary
    """
    x, y, c = next(iter(valid_loader))
    layer = csvae.decoderY[0]
    a = layer.weight # (1, 2)
    b = layer.bias # (1,)
    with torch.no_grad():
        z = csvae.posteriorZ(x).sample() # (bs, 2)
        w = csvae.posteriorW(x).sample() # (bs, 2)
        alpha = -(w @ a.view(2, 1) + b)/(a @ a.view(2, 1)) # (bs, 1)
        dw = step*alpha*a # (bs, 2)
        w_CF = w + dw
        x_CF = csvae.decode(w_CF, z)

    fig = plt.figure()

    ax = plt.subplot(3, 4, 1)
    plot_samples(ax, x[idx].view(1, -1), color=True)
    ax.set_title('Original')
    ax = plt.subplot(3, 4, 4)
    plot_samples(ax, x_CF[idx].view(1, -1), color=True)
    ax.set_title('Counterfactual')

    ax = plt.subplot(2, 1, 2)
    with torch.no_grad():
        for x, y, c in train_loader:
            w = csvae.posteriorW(x).sample()
            plot_scatter(ax, w, y)
    ax.set_xlabel('$w_0$')
    ax.set_ylabel('$w_1$')
    ax.arrow(w[0, 0], w[0, 1], dw[0, 0], dw[0, 1], head_width=1)

    plt.savefig(path + f'projection-{idx}-{step}.png')
    plt.close(fig)

def visualize_reconstructions(data_loader, csvae, path, figsize = (15, 5), color=False):
    x, y = next(iter(data_loader))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].set_title('Observations')
    plot_samples(axes[0], x, color)

    with torch.no_grad():
        w = csvae.posteriorW(x).sample()
        z = csvae.posteriorZ(x).sample()
        l_CF = csvae.decode(w, z)
        px_CF = csvae.observation_model(l_CF)
        x_CF = px_CF.mean

    axes[1].set_title('Reconstructions using the CB parameters')
    plot_samples(axes[1], l_CF, color)

    axes[2].set_title('Reconstructions using the CB mean')
    plot_samples(axes[2], x_CF, color)

    plt.savefig(path + 'reconstructions.png')
    plt.close(fig)

def get_data_ranges(x):
    '''
    Determine minimum and maximum for each feature in input x and output as 
    numpy array.
    
    Args:
            x:          An array of shape (N,M), where M corresponds to 
                        features and N corresponds to observations.
                        
    Returns:
            ranges:     A numpy array of minimum and maximum values for each  
                        feature dimension.
    '''
    N, M = x.shape
    ranges = []
    for m in range(M):
        ranges.append(np.min(x[:,m]))
        ranges.append(np.max(x[:,m]))
    return np.array(ranges)

def visualize_decision_boundary(predict, 
                                 X, y, 
                                 attribute_names,
                                 class_names,
                                 path,
                                 c = ['tab:orange', 'tab:blue'],
                                 train=None, test=None, 
                                 delta=5e-3,
                                 show_legend=False):
    '''
    Visualize the decision boundary of a classifier trained on a 2 dimensional
    input feature space.
    
    Creates a grid of points based on ranges of features in X, then determines
    classifier output for each point. The predictions are color-coded and plotted
    along with the data and a visualization of the partitioning in training and
    test if provided.
    
    Args:
        predict:
                A lambda function that takes the a grid of shape [M, N] as 
                input and returns the prediction of the classifier. M corre-
                sponds to the number of features (M==2 required), and N corre-
                sponding to the number of points in the grid. Can e.g. be a 
                trained PyTorch network (torch.nn.Sequential()), such as trained
                using toolbox_02450.train_neural_network, where the provided
                function would be something similar to: 
                >>> predict = lambda x: (net(torch.tensor(x, dtype=torch.float))).data.numpy()
                
        X:      A numpy array of shape (N, M), where N is the number of 
                observations and M is the number of input features (constrained
                to M==2 for this visualization).
                If X is a list of len(X)==2, then each element in X is inter-
                preted as a partition of training or test data, such that 
                X[0] is the training set and X[1] is the test set.
                
        y:      A numpy array of shape (N, 1), where N is the number of 
                observations. Each element is either 0 or 1, as the 
                visualization is constrained to a binary classification
                problem.
                If y is a list of len(y)==2, then each element in y is inter-
                preted as a partion of training or test data, such that 
                y[0] is the training set and y[1] is the test set. 
                
        attribute_names:
                A list of strings of length 2 giving the name
                of each of the M attributes in X.
                
        class_names: 
                A list of strings giving the name of each class in y.
                
        train (optional):  
                A list of indices describing the indices in X and y used for
                training the network. E.g. from the output of:
                    sklearn.model_selection.KFold(2).split(X, y)
                    
        test (optional):   
                A list of indices describing the indices in X and y used for
                testing the network (see also argument "train").
                
        delta (optional):
                A float describing the resolution of the decision
                boundary (default: 0.01). Default results grid of 100x100 that
                covers the first and second dimension range plus an additional
                25 percent.
        show_legend (optional):
                A boolean designating whether to display a legend. Defaults
                to True.
                
    Returns:
        Plots the decision boundary on a matplotlib.pyplot figure.
        
    '''
    plt.figure()    
    C = len(class_names)
    if isinstance(X, list) or isinstance(y, list):
        assert isinstance(y, list), 'If X is provided as list, y must be, too.'
        assert isinstance(y, list), 'If y is provided as list, X must be, too.'
        assert len(X)==2, 'If X is provided as a list, the length must be 2.'
        assert len(y)==2, 'If y is provided as a list, the length must be 2.'
        
        N_train, M = X[0].shape
        N_test, M = X[1].shape
        N = N_train+N_test
        grid_range = get_data_ranges(np.concatenate(X))
    else:
        N, M = X.shape
        grid_range = get_data_ranges(X)
    assert M==2, 'TwoFeatureError: Current neural_net_decision_boundary is only implemented for 2 features.'
    # Convert test/train indices to boolean index if provided:
    if train is not None or test is not None:
        assert not isinstance(X, list), 'Cannot provide indices of test and train partition, if X is provided as list of train and test partition.'
        assert not isinstance(y, list), 'Cannot provide indices of test and train partition, if y is provided as list of train and test partition.'
        assert train is not None, 'If test is provided, then train must also be provided.'
        assert test is not None, 'If train is provided, then test must also be provided.'
        train_index = np.array([(int(e) in train) for e in np.linspace(0, N-1, N)])
        test_index = np.array([(int(e) in test) for e in np.linspace(0, N-1, N)])
    
    xx = np.arange(grid_range[0], grid_range[1], delta)
    yy = np.arange(grid_range[2], grid_range[3], delta)
    # make a mesh-grid from a and b that spans the grid-range defined
    grid = np.stack(np.meshgrid(xx, yy))
    # reshape grid to be of shape "[number of feature dimensions] by [number of points in grid]"
    # this ensures that the shape fits the way the network expects input to be shaped
    # and determine estimated class label for entire featurespace by estimating
    # the label of each point in the previosly defined grid using provided
    # function predict()
    grid_predictions = predict(np.reshape(grid, (2,-1)).T)
    
    # Plot data with color designating class and transparency+shape
    # identifying partition (test/train)
    if C == 2:
        # c = ['tab:orange','tab:blue']
        cmap = cm.bwr
        vmax=1
    else:
        c = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
             'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        cmap = cm.tab10
        vmax=10
        
    s = ['o','x']; t = [.33, 1.0]
    
    if train is not None and test is not None:
        colors = [c[int(l)] for l in y]
        for (j, e) in enumerate([train_index, test_index]):
            plt.scatter(X[e, 0], X[e, 1], marker=s[j],c=colors[e], alpha=t[j])
    if isinstance(X,list) and isinstance(y, list):
        for (j, (X_par, y_par)) in enumerate(zip(X,y)):
            colors = [c[int(l)] for l in y_par]
            plt.scatter(X_par[:, 0], X_par[:, 1], marker=s[j], c=colors, alpha=t[j])
  
    plt.xlim(grid_range[0:2])
    plt.ylim(grid_range[2:])
    plt.xlabel(attribute_names[0])
    plt.ylabel(attribute_names[1])

    # reshape the predictions for each point in the grid to be shaped like
    # an image that corresponds to the feature-scace using the ranges that
    # defined the grid (a and b)
    decision_boundary = np.reshape(grid_predictions, (len(yy), len(xx)))
    # display the decision boundary
    ax = plt.imshow(decision_boundary, cmap=cmap, 
           extent=grid_range, vmin=0, vmax=vmax, alpha=.33, origin='lower')
    plt.axis('auto')
    if C == 2:
        plt.contour(grid[0], grid[1], decision_boundary, levels=[.5])
        plt.colorbar(ax, fraction=0.046, pad=0.04);
    if show_legend:
        plt.legend([class_names[i]+' '+e for i in range(C) for e in ['train','test']])
        
    plt.savefig(path)
    plt.close()
