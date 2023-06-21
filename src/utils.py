import torch
from sklearn.metrics import accuracy_score
import numpy as np
from model import AUX, MMD
import torch.nn as nn
from matplotlib import pyplot as plt
    
def get_cmnist_accuracy(data_loader, classifier, device):
    targets = []
    preds = []
    with torch.no_grad():
        for x, y, c in data_loader:
            x = x.to(device)
            y_pred = classifier(x)
            targets += list(y.numpy())
            preds += list(y_pred.cpu().numpy())
    return accuracy_score(targets, preds)
    
def median_distance(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    return ((a_core - b_core).pow(2)/2).sqrt().median()

def median_heuristic(dataset, csvae, device, conditional):

    x = dataset['images']
    y = dataset['labels']
    c = dataset['colors']

    with torch.no_grad():
        outputs = csvae(x.to(device), y.to(device))

    w, z = [outputs[k] for k in ["w", "z"]]

    if conditional:
        lw0 = median_distance(w[(y == 0) & (c == 0)], w[(y == 0) & (c == 1)])
        lw1 = median_distance(w[(y == 1) & (c == 0)], w[(y == 1) & (c == 1)])
        lz0 = median_distance(z[(c == 0) & (y == 0)], z[(c == 0) & (y == 1)])
        lz1 = median_distance(z[(c == 1) & (y == 0)], z[(c == 1) & (y == 1)])
        return lw0, lw1, lz0, lz1
    else:
        lw = median_distance(w[c == 0], w[c == 1])
        lz = median_distance(z[y == 0], z[y == 1])
        return lw, lz

def get_adversarial_error(csvae, lr, device, train_loader, valid_loader, test_loader, path, num_epochs = 10, validation_every_steps = 500):
    # aux_y = Adversarial(z_dim=3)
    # aux_c = Adversarial(z_dim=3)
    aux_y = AUX(z_dim=3)
    aux_c = AUX(z_dim=3)
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

            zc = torch.cat((z, c.view(-1, 1)), dim=1)
            wy = torch.cat((w, y.view(-1, 1)), dim=1)
            
            # output_y = aux_y(zc)
            # loss_y = loss_fn(output_y, y.long())
            qy = aux_y(zc)
            log_qy = qy.log_prob(y)
            aux_y_loss = log_qy.mean()
            optimizer_y.zero_grad()
            aux_y_loss.backward()
            optimizer_y.step()

            # output_c = aux_y(wy)
            # loss_c = loss_fn(output_c, c.long())
            qc = aux_c(wy)
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

                        zc = torch.cat((z, c.view(-1, 1)), dim=1)
                        wy = torch.cat((w, y.view(-1, 1)), dim=1)

                        # output_y = aux_y(zc)
                        # loss_y = loss_fn(output_y, y.long())
                        # predictions_y = output_y.max(1)[1]

                        # output_c = aux_c(wy)
                        # loss_c = loss_fn(output_c, c.long())
                        # predictions_c = output_c.max(1)[1]

                        # # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        # valid_accuracies_batches_y.append(accuracy_score(y.detach().cpu().numpy(), predictions_y.detach().cpu().numpy()) * len(y))
                        # valid_accuracies_batches_c.append(accuracy_score(c.detach().cpu().numpy(), predictions_c.detach().cpu().numpy()) * len(c))

                        qy = aux_y(zc)
                        log_qy = qy.log_prob(y)

                        qc = aux_c(wy)
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

            zc = torch.cat((z, c.view(-1, 1)), dim=1)
            wy = torch.cat((w, y.view(-1, 1)), dim=1)
            
            # output_y = aux_y(zc)
            # loss_y = loss_fn(output_y, y.long())
            # predictions_y = output_y.max(1)[1]

            # output_c = aux_c(wy)
            # loss_c = loss_fn(output_c, c.long())
            # predictions_c = output_c.max(1)[1]

            # # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=True).
            # test_accuracies_y.append(accuracy_score(y.cpu().numpy(), predictions_y.cpu().numpy()) * len(y))
            # test_accuracies_c.append(accuracy_score(c.cpu().numpy(), predictions_c.cpu().numpy()) * len(c))

            qy = aux_y(zc)
            log_qy = qy.log_prob(y)

            qc = aux_c(wy)
            log_qc = qc.log_prob(c)

            test_accuracies_y.append(torch.exp(log_qy).mean().item()*len(y))
            test_accuracies_c.append(torch.exp(log_qc).mean().item()*len(c))

        test_accuracy_y = np.sum(test_accuracies_y) / n_test
        test_accuracy_c = np.sum(test_accuracies_c) / n_test
        return test_accuracy_y, test_accuracy_c

def get_adversarial_posterior(csvae, lr, device, train_loader, valid_loader, test_loader, path, num_epochs = 10, validation_every_steps = 500):

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

def train_marg_adversary(net, csvae, lr, device, train_loader, valid_loader, path, tag='nl', num_epochs = 20, validation_every_steps = 500):

    aux_y = net(z_dim=2)
    aux_c = net(z_dim=2)
    aux_y = aux_y.to(device)
    aux_c = aux_c.to(device)

    optimizer_y = torch.optim.Adam(aux_y.parameters(), lr=lr)
    optimizer_c = torch.optim.Adam(aux_c.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_valid = len(valid_loader.dataset)
    
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
            
            output_y = aux_y(z)
            loss_y = loss_fn(output_y, y.long())
            optimizer_y.zero_grad()
            loss_y.backward()
            optimizer_y.step()

            output_c = aux_c(w)
            loss_c = loss_fn(output_c, c.long())
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()
            
            step += 1
            
            # Compute accuracy.
            predictions_y = output_y.max(1)[1]
            train_accuracies_batches_y.append(accuracy_score(y.detach().cpu().numpy(), predictions_y.cpu().numpy()))

            predictions_c = output_c.max(1)[1]
            train_accuracies_batches_c.append(accuracy_score(c.cpu().numpy(), predictions_c.cpu().numpy()))
            
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

                        output_y = aux_y(z)
                        loss_y = loss_fn(output_y, y.long())
                        predictions_y = output_y.max(1)[1]

                        output_c = aux_c(w)
                        loss_c = loss_fn(output_c, c.long())
                        predictions_c = output_c.max(1)[1]

                        # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        valid_accuracies_batches_y.append(accuracy_score(y.detach().cpu().numpy(), predictions_y.detach().cpu().numpy()) * len(y))
                        valid_accuracies_batches_c.append(accuracy_score(c.detach().cpu().numpy(), predictions_c.detach().cpu().numpy()) * len(c))

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
    plt.savefig(f'{path}/acc_y_{tag}.png')

    plt.figure()
    plt.plot(steps, train_accuracies_c, label='train')
    plt.plot(steps, valid_accuracies_c, label='validation')
    plt.xlabel('Training steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Train and validation accuracy for predicting c")
    plt.savefig(f'{path}/acc_c_{tag}.png')

    torch.save(aux_y.state_dict(), f'{path}/adv_y_{tag}.pt')
    torch.save(aux_c.state_dict(), f'{path}/adv_c_{tag}.pt')

    return aux_y, aux_c

def eval_marg_adversary(csvae, aux_y, aux_c, test_loader, device):
    # Evaluate test set
    loss_fn = nn.CrossEntropyLoss()
    n_test = len(test_loader.dataset)
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
            
            output_y = aux_y(z)
            loss_y = loss_fn(output_y, y.long())
            predictions_y = output_y.max(1)[1]

            output_c = aux_c(w)
            loss_c = loss_fn(output_c, c.long())
            predictions_c = output_c.max(1)[1]

            # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=True).
            test_accuracies_y.append(accuracy_score(y.cpu().numpy(), predictions_y.cpu().numpy()) * len(y))
            test_accuracies_c.append(accuracy_score(c.cpu().numpy(), predictions_c.cpu().numpy()) * len(c))

        test_accuracy_y = np.sum(test_accuracies_y) / n_test
        test_accuracy_c = np.sum(test_accuracies_c) / n_test
        return test_accuracy_y, test_accuracy_c

def train_cond_adversary(net, csvae, lr, device, train_loader, valid_loader, path, tag='nl', num_epochs = 20, validation_every_steps = 500):

    aux_y = net(z_dim=3)
    aux_c = net(z_dim=3)
    aux_y = aux_y.to(device)
    aux_c = aux_c.to(device)

    optimizer_y = torch.optim.Adam(aux_y.parameters(), lr=lr)
    optimizer_c = torch.optim.Adam(aux_c.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_valid = len(valid_loader.dataset)
    
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

            zc = torch.cat((z, c.view(-1, 1)), dim=1)
            wy = torch.cat((w, y.view(-1, 1)), dim=1)
            
            output_y = aux_y(zc)
            loss_y = loss_fn(output_y, y.long())
            optimizer_y.zero_grad()
            loss_y.backward()
            optimizer_y.step()

            output_c = aux_c(wy)
            loss_c = loss_fn(output_c, c.long())
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()
            
            step += 1
            
            # Compute accuracy.
            predictions_y = output_y.max(1)[1]
            train_accuracies_batches_y.append(accuracy_score(y.detach().cpu().numpy(), predictions_y.cpu().numpy()))

            predictions_c = output_c.max(1)[1]
            train_accuracies_batches_c.append(accuracy_score(c.cpu().numpy(), predictions_c.cpu().numpy()))
            
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

                        zc = torch.cat((z, c.view(-1, 1)), dim=1)
                        wy = torch.cat((w, y.view(-1, 1)), dim=1)

                        output_y = aux_y(zc)
                        loss_y = loss_fn(output_y, y.long())
                        predictions_y = output_y.max(1)[1]

                        output_c = aux_c(wy)
                        loss_c = loss_fn(output_c, c.long())
                        predictions_c = output_c.max(1)[1]

                        # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        valid_accuracies_batches_y.append(accuracy_score(y.detach().cpu().numpy(), predictions_y.detach().cpu().numpy()) * len(y))
                        valid_accuracies_batches_c.append(accuracy_score(c.detach().cpu().numpy(), predictions_c.detach().cpu().numpy()) * len(c))

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
    plt.savefig(f'{path}/acc_y_{tag}.png')

    plt.figure()
    plt.plot(steps, train_accuracies_c, label='train')
    plt.plot(steps, valid_accuracies_c, label='validation')
    plt.xlabel('Training steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Train and validation accuracy for predicting c")
    plt.savefig(f'{path}/acc_c_{tag}.png')

    torch.save(aux_y.state_dict(), f'{path}/adv_y_{tag}.pt')
    torch.save(aux_c.state_dict(), f'{path}/adv_c_{tag}.pt')

    return aux_y, aux_c

def eval_cond_adversary(csvae, aux_y, aux_c, test_loader, device):
    # Evaluate test set
    loss_fn = nn.CrossEntropyLoss()
    n_test = len(test_loader.dataset)
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

            zc = torch.cat((z, c.view(-1, 1)), dim=1)
            wy = torch.cat((w, y.view(-1, 1)), dim=1)
            
            output_y = aux_y(zc)
            loss_y = loss_fn(output_y, y.long())
            predictions_y = output_y.max(1)[1]

            output_c = aux_c(wy)
            loss_c = loss_fn(output_c, c.long())
            predictions_c = output_c.max(1)[1]

            # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=True).
            test_accuracies_y.append(accuracy_score(y.cpu().numpy(), predictions_y.cpu().numpy()) * len(y))
            test_accuracies_c.append(accuracy_score(c.cpu().numpy(), predictions_c.cpu().numpy()) * len(c))

        test_accuracy_y = np.sum(test_accuracies_y) / n_test
        test_accuracy_c = np.sum(test_accuracies_c) / n_test
        return test_accuracy_y, test_accuracy_c

def train_adversary_dual(net, csvae, lr, device, train_loader, valid_loader, path, tag='nl', num_epochs = 20, validation_every_steps = 500):

    aux_y0 = net(z_dim=2)
    aux_c0 = net(z_dim=2)
    aux_y0 = aux_y0.to(device)
    aux_c0 = aux_c0.to(device)

    aux_y1 = net(z_dim=2)
    aux_c1 = net(z_dim=2)
    aux_y1 = aux_y1.to(device)
    aux_c1 = aux_c1.to(device)

    optimizer_y0 = torch.optim.Adam(aux_y0.parameters(), lr=lr)
    optimizer_c0 = torch.optim.Adam(aux_c0.parameters(), lr=lr)
    optimizer_y1 = torch.optim.Adam(aux_y1.parameters(), lr=lr)
    optimizer_c1 = torch.optim.Adam(aux_c1.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_valid = len(valid_loader.dataset)
    
    step = 0
    aux_y_0.train()
    aux_c_0.train()
    aux_y_1.train()
    aux_c_1.train()

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

            z0 = z[c == 0]
            z1 = z[c == 1]
            w0 = w[y == 0]
            w1 = w[y == 1]
            y0 = y[c == 0]
            y1 = y[c == 1]
            c0 = c[y == 0]
            c1 = c[y == 1]
            nc0 = sum(c==0)
            nc1 = sum(c==1)
            ny0 = sum(y==0)
            ny1 = sum(y==1)
            
            output_y_0 = aux_y0(z0)
            output_y_1 = aux_y1(z1)
            loss_y = (nc0*loss_fn(output_y0, y0.long()) + nc1*loss_fn(output_y1, y1.long()))/(nc0 + nc1)
            optimizer_y.zero_grad()
            loss_y.backward()
            optimizer_y.step()

            output_c0 = aux_c0(w0)
            output_c1 = aux_c1(w1)
            loss_c = (ny0*loss_fn(output_c0, c0.long()) + ny1*loss_fn(output_c1, c1.long()))/(ny0 + ny1)
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()
            
            step += 1
            
            # Compute accuracy.
            predictions_y = output_y.max(1)[1]
            train_accuracies_batches_y.append(accuracy_score(y.detach().cpu().numpy(), predictions_y.cpu().numpy()))

            predictions_c = output_c.max(1)[1]
            train_accuracies_batches_c.append(accuracy_score(c.cpu().numpy(), predictions_c.cpu().numpy()))
            
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

                        z0 = z[c == 0]
                        z1 = z[c == 1]
                        w0 = w[y == 0]
                        w1 = w[y == 1]
                        y0 = y[c == 0]
                        y1 = y[c == 1]
                        c0 = c[y == 0]
                        c1 = c[y == 1]
                        nc0 = sum(c==0)
                        nc1 = sum(c==1)
                        ny0 = sum(y==0)
                        ny1 = sum(y==1)

                        output_y0 = aux_y0(z)
                        loss_y0 = loss_fn(output_y0, y0.long())
                        predictions_y0 = output_y0.max(1)[1]

                        output_c0 = aux_c0(w)
                        loss_c0 = loss_fn(output_c0, c0.long())
                        predictions_c0 = output_c0.max(1)[1]

                        output_y1 = aux_y1(z)
                        loss_y1 = loss_fn(output_y1, y1.long())
                        predictions_y1 = output_y1.max(1)[1]

                        output_c1 = aux_c1(w)
                        loss_c1 = loss_fn(output_c1, c1.long())
                        predictions_c1 = output_c1.max(1)[1]

                        accuracy_y0 = accuracy_score(y0.detach().cpu().numpy(), predictions_y0.detach().cpu().numpy())
                        accuracy_y1 = accuracy_score(y1.detach().cpu().numpy(), predictions_y1.detach().cpu().numpy())
                        accuracy_c0 = accuracy_score(c0.detach().cpu().numpy(), predictions_c0.detach().cpu().numpy())
                        accuracy_c1 = accuracy_score(c1.detach().cpu().numpy(), predictions_c1.detach().cpu().numpy())

                        # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        valid_accuracies_batches_y.append(ny0*accuracy_y0 + ny1*accuracy_y1)
                        valid_accuracies_batches_c.append(nc0*accuracy_c0 + nc1*accuracy_c1)

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
    plt.savefig(f'{path}/acc_y_{tag}.png')

    plt.figure()
    plt.plot(steps, train_accuracies_c, label='train')
    plt.plot(steps, valid_accuracies_c, label='validation')
    plt.xlabel('Training steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Train and validation accuracy for predicting c")
    plt.savefig(f'{path}/acc_c_{tag}.png')

    torch.save(aux_y.state_dict(), f'{path}/adv_y_{tag}.pt')
    torch.save(aux_c.state_dict(), f'{path}/adv_c_{tag}.pt')

    return aux_y, aux_c

def eval_adversary_dual(csvae, aux_y0, aux_c0, aux_y1, aux_c1, test_loader, device):
    # Evaluate test set
    loss_fn = nn.CrossEntropyLoss()
    n_test = len(test_loader.dataset)
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
            
            z0 = z[c == 0]
            z1 = z[c == 1]
            w0 = w[y == 0]
            w1 = w[y == 1]
            y0 = y[c == 0]
            y1 = y[c == 1]
            c0 = c[y == 0]
            c1 = c[y == 1]
            nc0 = sum(c==0)
            nc1 = sum(c==1)
            ny0 = sum(y==0)
            ny1 = sum(y==1)

            output_y0 = aux_y0(z)
            loss_y0 = loss_fn(output_y0, y0.long())
            predictions_y0 = output_y0.max(1)[1]

            output_c0 = aux_c0(w)
            loss_c0 = loss_fn(output_c0, c0.long())
            predictions_c0 = output_c0.max(1)[1]

            output_y1 = aux_y1(z)
            loss_y1 = loss_fn(output_y1, y1.long())
            predictions_y1 = output_y1.max(1)[1]

            output_c1 = aux_c1(w)
            loss_c1 = loss_fn(output_c1, c1.long())
            predictions_c1 = output_c1.max(1)[1]

            accuracy_y0 = accuracy_score(y0.detach().cpu().numpy(), predictions_y0.detach().cpu().numpy())
            accuracy_y1 = accuracy_score(y1.detach().cpu().numpy(), predictions_y1.detach().cpu().numpy())
            accuracy_c0 = accuracy_score(c0.detach().cpu().numpy(), predictions_c0.detach().cpu().numpy())
            accuracy_c1 = accuracy_score(c1.detach().cpu().numpy(), predictions_c1.detach().cpu().numpy())

            # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
            valid_accuracies_batches_y.append(ny0*accuracy_y0 + ny1*accuracy_y1)
            valid_accuracies_batches_c.append(nc0*accuracy_c0 + nc1*accuracy_c1)

        test_accuracy_y = np.sum(test_accuracies_y) / n_test
        test_accuracy_c = np.sum(test_accuracies_c) / n_test
        return test_accuracy_y, test_accuracy_c

def get_conditional_mmd(dataset, csvae, device):
    lw0, lw1, lz0, lz1 = median_heuristic(dataset, csvae, device, conditional=True)
    x = dataset['images']
    y = dataset['labels']
    c = dataset['colors']
    with torch.no_grad():
        outputs = csvae(x.to(device), y.to(device))
    w, z = [outputs[k] for k in ["w", "z"]]
    mmd_w0 = MMD(w[(y == 0) & (c == 0)], w[(y == 0) & (c == 1)], lw0).item()
    mmd_w1 = MMD(w[(y == 1) & (c == 0)], w[(y == 1) & (c == 1)], lw1).item()
    mmd_z0 = MMD(z[(c == 0) & (y == 0)], z[(c == 0) & (y == 1)], lz0).item()
    mmd_z1 = MMD(z[(c == 1) & (y == 0)], z[(c == 1) & (y == 1)], lz1).item()
    return mmd_w0, mmd_w1, mmd_z0, mmd_z1

def get_color_switching_ratio(dataloader, csvae, device, step=2, thresh=0.01):
    n_switch = 0
    n_appear = 0
    with torch.no_grad():
        for x, y, c in dataloader:
            x = x.to(device)
            c = c.long()
            batch_size = x.shape[0]
            w = csvae.posteriorW(x).sample()            
            layer = csvae.decoderY[0]
            a = layer.weight 
            b = layer.bias 
            z = csvae.posteriorZ(x).sample() 
            w = csvae.posteriorW(x).sample() 
            alpha = -(w @ a.view(2, 1) + b)/(a @ a.view(2, 1)) 
            dw = step*alpha*a 
            w_CF = w + dw
            x_CF = csvae.decode(w_CF, z).view(-1, 2, 14*14)
            color_sum = x_CF.sum(axis=2)
            for i in range(batch_size):
                if color_sum[i, c[i]] - color_sum[i, 1-c[i]] < 0:
                    n_switch += 1
                if sum(x_CF[i, 1 - c[i]] > thresh) > 0:
                    n_appear += 1
    return n_switch, n_appear

def eval_by(val_loader, csvae, vi, device):
    log_px = []
    qy = []
    with torch.no_grad():
        csvae.eval()
        # Go through each batch in the training dataset using the loader
        # Note that y is not necessarily known as it is here
        for x, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            # perform a forward pass through the model and compute the ELBO
            loss, diagnostics, outputs = vi(csvae, x, y)
            
            # gather data for the current batch
            log_px.append(diagnostics['log_px'].mean().item())
            qy.append(diagnostics['qy'].mean().item())
            
    return np.mean(log_px), np.mean(qy)    