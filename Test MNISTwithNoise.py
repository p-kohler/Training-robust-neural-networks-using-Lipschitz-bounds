# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:21:54 2020

@author: paul_
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torchsummary import summary
from scipy.io import savemat
import numpy as np
import os
import solve_SDP_multi
import matplotlib.pyplot as plt
import time
import matlab.engine
import scipy.io
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


INPUT_SIZE = 196  # 784
HIDDEN_SIZE = 50
OUTPUT_SIZE = 10
BATCH_SIZE = 100
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
c = 0.05  # robustness parameter for LMT
rho = 0.25  # ADMM penalty parameter
mu = 0.01  # Lip penalty parameter
lmbd = 0.0005  # L2 penalty parameter
it_ADMM = 10
ind_Lip = 1  # 1 Lipschitz regularization, 2 Enforcing Lipschitz bounds
Lip_nom = 10

net_dims = [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]


class Network(nn.Module):
    def __init__(self, activation=torch.tanh):
        """Constructor for multi-layer perceptron pytorch class
        params:
            * net_dims: list of ints  - dimensions of each layer in neural network
            * activation: func        - activation function to use in each layer
                                      - default is ReLU
        """
        super(Network, self).__init__()


    def test_model(self, test_loader):
        """Test neural network model using argmax classification
        params:
            * model: nn.Sequential instance   - torch NN model to be tested
            * test_loader:                    - Test data for NN
        returns:
            * test_accuracy: float - testing classification accuracy
        """
        self.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.view(BATCH_SIZE, -1)
                output = self(data)

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)               # Increment the total count
                correct += (predicted == labels).sum()     # Increment the correct count

        test_accuracy = 100 * correct.numpy() / float(total)
        print('Test Accuracy: %.3f %%\n' % test_accuracy)

        return test_accuracy


def create_data_loaders(std = None, b = None):
    """Create DataLoader instances for training and testing neural networks
    returns:
        * train_loader: DataLoader instance   - loader for training set
        * test_loader: DataLoader instance    - loader for test set
    """
    train_set = datasets.MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

    if std is not None:
        test_set = datasets.MNIST('/tmp', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               AddGaussianNoise(0., std)]))
    elif b is not None:
        test_set = datasets.MNIST('/tmp', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               AddUniformNoise(0., b)]))
        
    elif: 
        test_set = datasets.MNIST('/tmp', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               AddUniformNoise(0., b)]))
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddUniformNoise(object):
    def __init__(self, a=0., b=1.):
        self.a = a
        self.b = b

    def __call__(self, tensor):
        return tensor + (torch.rand(tensor.size()) - 0.5) * self.b + self.a

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def main():

    # Werte anzeigen mit b=0
    train_loader, test_loader = create_data_loaders(0)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data[0][0])


    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig


    # Werte anzigen mit b=2
    train_loader, test_loader = create_data_loaders(2)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data[0][0])

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig

    modelNom = torch.load('MNIST196_NomModel.pt')
    modelL2 = torch.load('MNIST196_L2Model.pt')
    modelLip = torch.load('MNIST196_LipModel.pt')
    modelLMT = torch.load('MNIST196_LMTModel.pt')

    std_array = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    b_array = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2]

    for i in range(len(b_array)):

        std = std_array[i]
        
        train_loader, test_loader = create_data_loaders(std)
        
        print('std = {}'.format(std))

        print('Nominal NN:')
        modelNom.test_model(test_loader)
        print('L2 NN:')
        modelL2.test_model(test_loader)
        print('Lipschitz NN:')
        modelLip.test_model(test_loader)
        print('LMT NN:')
        modelLMT.test_model(test_loader)
        

    for i in range(len(b_array)):

        b = b_array[i]

        train_loader, test_loader = create_data_loaders(b)

        print('b = {}'.format(b))

        print('Nominal NN:')
        modelNom.test_model(test_loader)
        print('L2 NN:')
        modelL2.test_model(test_loader)
        print('Lipschitz NN:')
        modelLip.test_model(test_loader)
        print('LMT NN:')
        modelLMT.test_model(test_loader)


if __name__ == '__main__':
    main()

