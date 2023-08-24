import torch
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split

def make_subset(dataset, classes):
    classDict = dict(zip(classes, range(len(classes))))

    N = len(dataset)
    indices = torch.zeros(N, dtype = bool)
    for c in classes:
        idx = (dataset.targets == c)
        indices = indices | idx

    return {
       'images': dataset.data[indices].float().view(-1, 784)/255.,
       'labels': torch.FloatTensor([classDict[cl.item()] for cl in dataset.targets[indices]])
       }

def make_mnist(data_path, classes = [4, 9]):
    """
    Save train and test datasets for MNIST with only a subset of the classes
    """
    dset_train = MNIST(data_path, train=True,  download=False)
    dset_test  = MNIST(data_path, train=False, download=False)
    mnist_train = make_subset(dset_train, classes)
    mnist_test = make_subset(dset_test, classes)
    x_train = mnist_train['images']
    y_train = mnist_train['labels']
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=len(mnist_test['labels']), stratify=y_train, random_state=0)
    mnist_train = {'images': x_train, 'labels': y_train}
    mnist_valid = {'images': x_valid, 'labels': y_valid}
    torch.save(mnist_train, f'{data_path}/mnist_train.pt')
    torch.save(mnist_valid, f'{data_path}/mnist_valid.pt')
    torch.save(mnist_test, f'{data_path}/mnist_test.pt')
   
if __name__ == "__main__":
    data_path = "./data"
    dset_train = MNIST(data_path, train=True, download=True)
    dset_test  = MNIST(data_path, train=False, download=True)
    make_mnist(data_path)
    