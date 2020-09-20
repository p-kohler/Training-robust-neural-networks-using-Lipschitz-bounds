# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:12:29 2020

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


INPUT_SIZE = 196  # 784
HIDDEN_SIZE = 50
HIDDEN_SIZE2 = 50
OUTPUT_SIZE = 10
net_dims = [INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, OUTPUT_SIZE]

BATCH_SIZE = 100
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
c = 0.05  # robustness parameter for LMT
rho = 0.25  # ADMM penalty parameter
mu = 0.01  # Lip penalty parameter
lmbd = 0.0005  # L2 penalty parameter
ind_Lip = 1  # 1 Lipschitz regularization, 2 Enforcing Lipschitz bounds
Lip_nom = 10


class Network(nn.Module):
    def __init__(self, activation=torch.tanh):
        """Constructor for multi-layer perceptron pytorch class
        params:
            * net_dims: list of ints  - dimensions of each layer in neural network
            * activation: func        - activation function to use in each layer
                                      - default is ReLU
        """
        super(Network, self).__init__()

        self.AvgPool = torch.nn.AvgPool2d(kernel_size=2)

        layers = []
        for i in range(len(net_dims) - 1):
            layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))

            # use activation function if not at end of layer
            if i != len(net_dims) - 2:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Pass data through the neural network model
        params:
            * x: torch tensor - data to pass though neural network
        returns:
            * ouput of neural network
        """

        # print()
        # print('Input shape: ', x.size())

        reshape_output = torch.reshape(x, (100, 28, 28))
        # print('Shape after reshaping: ', reshape_output.size())

        pooling_output = self.AvgPool(reshape_output)
        # print('Shape after pooling: ', pooling_output.size())

        reshape2_output = torch.reshape(pooling_output, (100, 196))
        # print('Shape after reshaping: ', reshape2_output.size())

        x = self.net(reshape2_output)
        # print('Shape after net: ', x.size())

        return x

    def extract_weights(self):
        weights = []
        biases = []
        for param_tensor in self.state_dict():
            tensor = self.state_dict()[param_tensor].detach().numpy().astype(np.float64)

            if 'weight' in param_tensor:
                weights.append(tensor)
            if 'bias' in param_tensor:
                biases.append(tensor)
        return weights, biases

    def Lip_reg(self, rho, mu, parameters):
        Lip_loss = None
        i = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                W_bar = torch.tensor(parameters['W{:d}_bar'.format(i)])
                Y = torch.tensor(parameters['Y{:d}'.format(i)])
                if Lip_loss is None:
                    Lip_loss = rho/2 * torch.sum((param-W_bar)**2) + torch.trace(torch.matmul(Y.t(), (param-W_bar)))
                else:
                    Lip_loss = Lip_loss + rho/2 * torch.sum((param-W_bar)**2) + torch.trace(torch.matmul(Y.t(), (param-W_bar)))
                i += 1
        return Lip_loss

    def l2_reg(self, lmbd):
        reg_loss = None
        for param in self.parameters():
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param**2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2)**2
        return lmbd * reg_loss

    def LMT_reg(self, Lip_nom, data, c, output, target, parameters=None):
        if Lip_nom is None:
            self.train_model()

        for j in range(len(data)):
            if np.argmax(output.detach().numpy()[j, :]) == target[j]:
                for cl in range(10):
                    if target[j] == cl:
                        for k in range(10):
                            if cl != k:
                                output.detach().numpy()[j, k] += np.sqrt(2)*Lip_nom*c

    def train_model(self, train_loader, test_loader, optimizer, criterion, lmbd=None, rho=None, mu=None, parameters=None, c=None, Lip_nom=None):
        """Train neural network model with Adam optimizer for a single epoch
        params:
            * model: nn.Sequential instance                 - NN model to be tested
            * train_loader: DataLoader instance             - Training data for NN
            * optimizer: torch.optim instance               - Optimizer for NN
            * criterion: torch.nn.CrossEntropyLoss instance - Loss function
            * epoch_num: int                                - Number of current epoch
            * log_interval: int                             - interval to print output
        modifies:
            weights of neural network model instance
        """
        self.train()   # Set model to training mode
        Lip_course, loss_course, CEloss_course, accuracy_course = [], [], [], []
        lossM, loss_prev, loss = 0, 0, 0

        for epoch_num in range(5):
            epoch_loss = 0
            epoch_CEloss = 0
            for batch_id, (data, target) in enumerate(train_loader):
                data = data.view(BATCH_SIZE, -1)

                optimizer.zero_grad()   # Zero gradient buffers
                output = self(data)    # Pass data through the network
                loss = criterion(output, target)    # Calculate loss

                if lmbd is not None:
                    loss += self.l2_reg(lmbd)

                if rho is not None:
                    loss += self.Lip_reg(rho, mu, parameters)

                if c is not None:
                    self.LMT_reg(Lip_nom, data, c, output)

                loss.backward()     # Backpropagate
                optimizer.step()    # Update weights

                epoch_loss += loss.item()
                epoch_CEloss += criterion(output, target)
                
            lossM = epoch_loss/batch_id

            if np.mod(epoch_num, 5) == 0:
                weights, biases = self.extract_weights()
                Lip = solve_SDP_multi.build_T_multi(weights, biases, net_dims)
                L_W = np.linalg.norm(weights[0], 2)*np.linalg.norm(weights[1], 2)
                T = Lip["T"]
                epoch_LipM = Lip["Lipschitz"]
                epoch_lossM = epoch_loss/batch_id
                epoch_CElossM = epoch_CEloss/batch_id
                accuracy = self.test_model(test_loader)
                print('Train Epoch: {}; Loss: {:.6f}; Cross-Entropy Loss: {:.6f}; Lipschitz: {:.3f}; Trivial Lipschitz: {:.3f}; Test Accuracy: {:.3f}'.format(epoch_num, epoch_lossM, epoch_CElossM, epoch_LipM, L_W, accuracy))

                Lip_course.append(epoch_LipM)
                loss_course.append(epoch_lossM)
                CEloss_course.append(epoch_CElossM)
                accuracy_course.append(accuracy)

        while abs(loss_prev - lossM) >= 0.01:
            epoch_num += 1
            epoch_loss = 0
            epoch_CEloss = 0
            for batch_id, (data, target) in enumerate(train_loader):
                data = data.view(BATCH_SIZE, -1)

                optimizer.zero_grad()   # Zero gradient buffers
                output = self(data)    # Pass data through the network
                loss = criterion(output, target)    # Calculate loss

                if lmbd is not None:
                    loss += self.l2_reg(lmbd)

                if rho is not None:
                    loss += self.Lip_reg(rho, mu, parameters)

                if c is not None:
                    self.LMT_reg(Lip_nom, data, c, output)

                loss.backward()     # Backpropagate
                optimizer.step()    # Update weights

                epoch_loss += loss.item()
                epoch_CEloss += criterion(output, target)

            if np.mod(epoch_num, 5) == 0:
                weights, biases = self.extract_weights()
                Lip = solve_SDP_multi.build_T_multi(weights, biases, net_dims)
                L_W = np.linalg.norm(weights[0], 2)*np.linalg.norm(weights[1], 2)
                T = Lip["T"]
                epoch_LipM = Lip["Lipschitz"]
                epoch_lossM = epoch_loss/batch_id
                epoch_CElossM = epoch_CEloss/batch_id
                accuracy = self.test_model(test_loader)
                print('Train Epoch: {}; Loss: {:.6f}; Cross-Entropy Loss: {:.6f}; Lipschitz: {:.3f}; Trivial Lipschitz: {:.3f}; Test Accuracy: {:.3f}'.format(epoch_num, epoch_lossM, epoch_CElossM, epoch_LipM, L_W, accuracy))

                Lip_course.append(epoch_LipM)
                loss_course.append(epoch_lossM)
                CEloss_course.append(epoch_CElossM)
                accuracy_course.append(accuracy)
                loss_prev = lossM
                lossM = epoch_lossM

        return loss_course, CEloss_course, Lip_course, accuracy_course, T

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
        # print('Test Accuracy: %.3f %%\n' % test_accuracy)

        return test_accuracy


def train_network(model, train_loader, test_loader, lmbd=None, rho=None, mu=None, parameters=None, c=None, Lip_nom=None, T=None, L_des=None):
    """Train a neural network with Adam optimizer
    params:
        * model: torch.nn instance            - neural network model
        * train_loader: DataLoader instance   - train dataset loader
        * test_loader: DataLoader instance    - test dataset loader
    returns:
        * accuracy: float - accuracy of trained neural network
    """
    Lip_course, loss_course, CEloss_course, accuracy_course = [], [], [], []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if rho is not None:
        Lip_prev, Lip_now = 0, 0

        for i in range(2):
            print("Beginn ADMM Iteration # {:d}".format(i))
            print("Beginn Training")
            t1 = time.time()
            epoch_loss, epoch_CEloss, epoch_Lip, epoch_accuracy, T = model.train_model(train_loader, test_loader, optimizer, criterion, rho=rho, mu=mu, parameters=parameters)
            timeLipTrain = time.time() - t1
            print("Training Complete after {} seconds.".format(timeLipTrain))

            for k in range(len(epoch_Lip)):
                Lip_course.append(epoch_Lip[k])
                loss_course.append(epoch_loss[k])
                CEloss_course.append(epoch_CEloss[k])
            weightsLip, biasesLip = model.extract_weights()
            scipy.io.savemat('c:/tmp/Lipparameters.mat', parameters)

            print("Beginn parameter update step")
            t = time.time()
            for j in range(len(weightsLip)):
                parameters.update({
                    'W{:d}'.format(j): matlab.double(np.array(weightsLip, dtype=np.object)[j].tolist()),
                    })
            timeParameterUpdate = time.time() - t
            print("Parameter update step complete after {} seconds.".format(timeParameterUpdate))

            print("Beginn Solving SDP (Lipschitz and Y update step)")
            t = time.time()
            parameters = solve_SDP_multi.solve_SDP_multi(parameters, T, net_dims, rho, mu, ind_Lip, L_des)  # Lipschitz and Y update steps
            timeSolveSDP = time.time() - t
            print("Solving SDP (Lipschitz and Y update step) complete after {} seconds.".format(timeSolveSDP))
            timeADMM = time.time() - t1
            print("ADMM Iteration # {:d} complete after {} seconds.".format(i, timeADMM))

            accuracy = model.test_model(test_loader)
            accuracy_course.append(accuracy)

            Lip_prev = Lip_now
            Lip_now = epoch_Lip[len(epoch_Lip)-1]

            scipy.io.savemat('c:/tmp/Lipparameters_updated.mat', parameters)

        while abs(Lip_prev - Lip_now) >= 0.5:
            i += 1
            print("Beginn ADMM Iteration # {:d}".format(i))
            print("Beginn Training")
            t1 = time.time()
            epoch_loss, epoch_CEloss, epoch_Lip, epoch_accuracy, T = model.train_model(train_loader, test_loader, optimizer, criterion, rho=rho, mu=mu, parameters=parameters)
            timeLipTrain = time.time() - t1
            print("Training Complete after {} seconds.".format(timeLipTrain))

            for k in range(len(epoch_Lip)):
                Lip_course.append(epoch_Lip[k])
                loss_course.append(epoch_loss[k])
                CEloss_course.append(epoch_CEloss[k])
            weightsLip, biasesLip = model.extract_weights()
            scipy.io.savemat('c:/tmp/Lipparameters.mat', parameters)

            print("Beginn parameter update step")
            t = time.time()
            for j in range(len(weightsLip)):
                parameters.update({
                    'W{:d}'.format(j): matlab.double(np.array(weightsLip, dtype=np.object)[j].tolist()),
                    })
            timeParameterUpdate = time.time() - t
            print("Parameter update step complete after {} seconds.".format(timeParameterUpdate))

            print("Beginn Solving SDP (Lipschitz and Y update step)")
            t = time.time()
            parameters = solve_SDP_multi.solve_SDP_multi(parameters, T, net_dims, rho, mu, ind_Lip, L_des)  # Lipschitz and Y update steps
            timeSolveSDP = time.time() - t
            print("Solving SDP (Lipschitz and Y update step) complete after {} seconds.".format(timeSolveSDP))
            timeADMM = time.time() - t1
            print("ADMM Iteration # {:d} complete after {} seconds.".format(i, timeADMM))

            accuracy = model.test_model(test_loader)
            accuracy_course.append(accuracy)

            Lip_prev = Lip_now
            Lip_now = epoch_Lip[len(epoch_Lip)-1]

            scipy.io.savemat('c:/tmp/Lipparameters_updated.mat', parameters)

    if lmbd is not None:
        loss_course, CEloss_course, Lip_course, accuracy_course, T = model.train_model(train_loader, test_loader, optimizer, criterion, lmbd)
        parameters = None
    elif c is not None:
        loss_course, CEloss_course, Lip_course, accuracy_course, T = model.train_model(train_loader, test_loader, optimizer, criterion, c)
        parameters = None
    elif rho is None and c is None and lmbd is None:
        loss_course, CEloss_course, Lip_course, accuracy_course, T = model.train_model(train_loader, test_loader, optimizer, criterion)
        parameters = None

    return parameters, Lip_course, loss_course, CEloss_course, accuracy_course


def create_data_loaders():
    """Create DataLoader instances for training and testing neural networks
    returns:
        * train_loader: DataLoader instance   - loader for training set
        * test_loader: DataLoader instance    - loader for test set
    """
    train_set = datasets.MNIST('/tmp', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

    test_set = datasets.MNIST('/tmp', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def main():

    train_loader, test_loader = create_data_loaders()
    print(train_loader)
    print(test_loader)
    fname = os.path.join(os.getcwd(), 'saved_weights/mnist_weights.mat')

    # nom NN

    # define neural network model and print summary
    net_dims = [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = Network(activation=nn.ReLU).to(device)
    summary(model, (50, 28, 28))

    # train model
    print("Beginnning nominal NN training")
    t = time.time()
    parametersNom, Lip_course, loss_course, CEloss_course, accuracy_course = train_network(model, train_loader, test_loader)
    timeNom = time.time() - t
    print("Nominal training complete after {} seconds".format(timeNom))

    # save data to saved_weights/ directory
    weights, biases = model.extract_weights()
    data = {'weights': np.array(weights, dtype=np.object)}
    savemat(fname, data)
    Lip_dic = solve_SDP_multi.build_T_multi(weights, biases, net_dims)
    Lip_nom = Lip_dic["Lipschitz"]
    torch.save(model, 'MNIST196_NomModel.pt')

    # plot losscourse
    plt.plot(loss_course)
    plt.xlabel('# epochs x 5')
    plt.ylabel('Loss Nom')
    plt.show()

    # plot CElosscourse
    plt.plot(CEloss_course)
    plt.xlabel('# epochs x 5')
    plt.ylabel('CE-Loss Nom')
    plt.show()

    # plot Lip_course
    plt.plot(Lip_course)
    plt.xlabel('# epochs x 5')
    plt.ylabel('Lip_course Nom')
    plt.show()

    # plot accuracy_course
    plt.plot(accuracy_course)
    plt.xlabel('# epochs x 5')
    plt.ylabel('accuracy_course Nom')
    plt.show()

    # L2 NN

    # define neural network model and print summary
    modelL2 = Network(activation=nn.ReLU).to(device)
    summary(model, (50, 28, 28))

    # train model
    print("Beginnning L2 training")
    t = time.time()
    parametersL2, Lip_courseL2, loss_courseL2, CEloss_courseL2, accuracy_courseL2 = train_network(modelL2, train_loader, test_loader, lmbd=lmbd)
    timeL2 = time.time() - t
    print("L2 training complete after {} seconds".format(timeL2))

    # save data to saved_weights/ directory
    weightsL2, biasesL2 = modelL2.extract_weights()
    data = {'weightsL2': np.array(weightsL2, dtype=np.object)}
    savemat(fname, data)
    Lip_L2 = solve_SDP_multi.build_T_multi(weightsL2, biasesL2, net_dims)
    torch.save(modelL2, 'MNIST196_L2Model.pt')

    # plot losscourse
    plt.plot(loss_courseL2)
    plt.xlabel('# epochs x 5')
    plt.ylabel('Loss L2')
    plt.show()

    # plot CElosscourse
    plt.plot(CEloss_courseL2)
    plt.xlabel('# epochs x 5')
    plt.ylabel('CE-Loss L2')
    plt.show()

    # plot Lip_course
    plt.plot(Lip_courseL2)
    plt.xlabel('# epochs x 5')
    plt.ylabel('Lip_course L2')
    plt.show()

    # plot accuracy_course
    plt.plot(accuracy_courseL2)
    plt.xlabel('# epochs x 5')
    plt.ylabel('accuracy_course L2')
    plt.show()

    # NN with Lipschitz regularizer

    rho_array = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 5]
    mu_array = [0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.1]

    for k in range(len(rho_array)):

        rho = rho_array[k]

        for m in range(len(mu_array)):

            mu = mu_array[m]

            # define neural network model and print summary
            modelLip = Network(activation=nn.ReLU).to(device)
            modelLip.load_state_dict(modelL2.state_dict())
            summary(model, (50, 28, 28))

            # train model
            L_des = Lip_L2["Lipschitz"]

            print("Beginnning parameters = solve_SDP1")
            t1 = time.time()
            parameters = solve_SDP_multi.initialize_parameters(weights, biases)
            timeSolveSDP1 = time.time() - t1
            print("Complete parameters = solve_SDP1 after {} seconds".format(timeSolveSDP1))
            print("Beginnning parameters = solve_SDP2")
            t2 = time.time()
            parameters_L2 = solve_SDP_multi.initialize_parameters(weightsL2, biasesL2)
            timeSolveSDP2 = time.time() - t2
            print("Complete parameters = solve_SDP2 after {} seconds".format(timeSolveSDP2))

            init = 1  # 1 initialize from L2-NN, 2 initialize from nominal NN
            if init == 1:
                print("Beginnning LipSDP training")
                t3 = time.time()
                parameters_Lip, Lip_courseLip, loss_courseLip, CEloss_courseLip, accuracy_courseLip = train_network(modelLip, train_loader, test_loader, rho=rho, mu=mu, parameters=parameters_L2, L_des=L_des, T=Lip_L2["T"])
                timeTrainSDP = time.time() - t3
                print("LipSDP training complete after {} seconds".format(timeTrainSDP))
            else:
                print("Beginnning LipSDP training")
                t3 = time.time()
                parameters_Lip, Lip_courseLip, loss_courseLip, CEloss_courseLip, accuracy_courseLip = train_network(modelLip, train_loader, test_loader, rho=rho, mu=mu, parameters=parameters, L_des=L_des, T=Lip_dic["T"])
                timeTrainSDP = time.time() - t3
                print("LipSDP training complete after {} seconds".format(timeTrainSDP))

            timeFullSDP = time.time() - t1
            print("Full LipSDP training complete after {} seconds".format(timeFullSDP))

            # save data to saved_weights/ directory
            weightsLip, biasesLip = modelLip.extract_weights()
            # weightsLip2, biasesLip2 = modelLip2.extract_weights()

            Lip_Lip = solve_SDP_multi.build_T_multi(weightsLip, biasesLip, net_dims)
            # Lip_Lip2 = solve_SDP.build_T(weightsLip2, biasesLip2, net_dims)
            data = {'weightsLip': np.array(weightsLip, dtype=np.object)}
            savemat(fname, data)
            torch.save(model, 'MNIST196_LipModel rho = {}, mu = {}.pt'.format(rho, mu))

            # plot losscourse
            plt.plot(loss_courseLip)
            plt.xlabel('# epochs x 5')
            plt.ylabel('Loss Lip')
            plt.title('Loss with rho = '+str(rho)+', mu = '+str(mu))
            plt.show()

            # plot CElosscourse
            plt.plot(CEloss_courseLip)
            plt.xlabel('# epochs x 5')
            plt.ylabel('CE-Loss Lip')
            plt.title('CE-Loss with rho = '+str(rho)+', mu = '+str(mu))
            plt.show()

            # plot Lip_course
            plt.plot(Lip_courseLip)
            plt.xlabel('# epochs x 5')
            plt.ylabel('Lip_course Lip')
            plt.title('Lip with rho = '+str(rho)+', mu = '+str(mu))
            plt.show()

            # plot accuracy_course
            plt.plot(accuracy_courseLip)
            plt.xlabel('# epochs x 5')
            plt.ylabel('accuracy_course Lip')
            plt.title('Accuracy with rho = '+str(rho)+', mu = '+str(mu))
            plt.show()


if __name__ == '__main__':
    main()
