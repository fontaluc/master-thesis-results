import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torch

dset_train = MNIST('../data', train=True,  download=True)
classes = torch.tensor([4, 9])
train_index = (dset_train.targets[:, None] == classes).any(-1).nonzero(as_tuple=True)[0]

images, labels = dset_train.data[train_index][:5], dset_train.targets[train_index][:5]
plt.imsave('outputs/figures/mnist.png', images[0], cmap='gray')
# 2x subsample for computational convenience
images = images.reshape((-1, 28, 28))[:, ::2, ::2]
plt.imsave('outputs/figures/mnist_sub.png', images[0], cmap='gray')
# Label 4 as 0 and 9 as 1
labels = (labels == 9).float()
colors = labels
# Apply the color to the image by zeroing out the other color channel
images1 = torch.stack([images, images], dim=1)
images1[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
image1 = torch.cat((images1[0].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0)
plt.imsave('outputs/figures/cmnist1.png', image1.to(torch.uint8).numpy())

colors = 1 - labels
# Apply the color to the image by zeroing out the other color channel
images2 = torch.stack([images, images], dim=1)
images2[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
image2 = torch.cat((images2[0].view(2, 14, 14), torch.zeros(1, 14, 14))).permute(1, 2, 0)
plt.imsave('outputs/figures/cmnist2.png', image2.to(torch.uint8).numpy())