# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 02:21:30 2020

@author: paul_
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import solve_SDP_multi
import matlab.engine
from datetime import datetime
from scipy.io import savemat


INPUT_SIZE = 2
HIDDEN_SIZE1 = 10
HIDDEN_SIZE2 = 10
OUTPUT_SIZE = 3
net_dims = [INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE]

# hyperperparameter
c = 0.05  # robustness for LMT
lr = 0.1  # learning rate
rho = 0.25  # ADMM penalty parameter
mu = 0.00001  # Lip penalty parameter
lmbd = 0.0005  # L2 penalty parameter
ind_Lip = 1  # 1 Lipschitz regularization, 2 Enforcing Lipschitz bounds


class MeinNetz(nn.Module):
    def __init__(self):
        super(MeinNetz, self).__init__()
        self.lin1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE1)
        self.lin2 = nn.Linear(HIDDEN_SIZE1, HIDDEN_SIZE2)
        self.lin3 = nn.Linear(HIDDEN_SIZE2, OUTPUT_SIZE)

    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        x = self.lin3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num

    def Lip_reg(self, rho, parameters):
        Lip_loss = None
        i = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                W_bar = torch.tensor(parameters['W{:d}_bar'.format(i)])
                Y = torch.tensor(parameters['Y{:d}'.format(i)])
                if Lip_loss is None:
                    Lip_loss =  rho/2 * torch.sum((param-W_bar)**2) + torch.trace(torch.matmul(Y.t(),(param-W_bar)))
                else:
                    Lip_loss = Lip_loss +  rho/2 * torch.sum((param-W_bar)**2) + torch.trace(torch.matmul(Y.t(),(param-W_bar)))
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

    def LMT_reg(self, c, parameters=None):
        if Lip_nom is None:
            Lip_course, loss_course, CEloss_course, T = self.train()

        for j in range(N):
            if np.argmax(out.detach().numpy()[j, :]) == target_cross[j]:
                if target_cross[j] == 0:
                    out.detach().numpy()[j, 1] += np.sqrt(2)*Lip_nom*c
                    out.detach().numpy()[j, 2] += np.sqrt(2)*Lip_nom*c
                if target_cross[j] == 1:
                    out.detach().numpy()[j, 0] += np.sqrt(2)*Lip_nom*c
                    out.detach().numpy()[j, 2] += np.sqrt(2)*Lip_nom*c
                if target_cross[j] == 2:
                    out.detach().numpy()[j, 0] += np.sqrt(2)*Lip_nom*c
                    out.detach().numpy()[j, 1] += np.sqrt(2)*Lip_nom*c

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

    def train(self, lmbd=None, rho=None, parameters=None, c=None):
        Lip_course = []
        loss_course = []
        CEloss_course = []
        out = self(input)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, target_cross)
        loss_prev = 0
        loss_prevprev = 0

        for i in range(10000):

            out = self(input)

            loss_prev = loss
            loss = criterion(out, target_cross)

            if lmbd is not None:
                loss += self.l2_reg(lmbd)

            if rho is not None:
                loss += self.Lip_reg(rho, parameters)

            if c is not None:
                self.LMT_reg(c)

            if np.mod(i, 5000) == 0:
                weights, biases = self.extract_weights()
                Lip = solve_SDP_multi.build_T_multi(weights, biases, net_dims)
                L = Lip["Lipschitz"]
                T = Lip["T"]
                L_W = 1
                for j in range(len(weights)):
                    L_W = L_W * np.linalg.norm(weights[j], 2)
                Lip_course.append(L)
                loss_course.append(loss.item())
                crossEntropyLoss = criterion(out, target_cross)
                CEloss_course.append(crossEntropyLoss)
                print('Train Epoch: {}; Loss: {:.6f}; CE-Loss: {:.6f}; Lipschitz: {:.3f}; Trivial Lipschitz: {:.3f}'.format(
                    i, loss.item(), crossEntropyLoss, L, L_W))
                # print(Lip["ok"])
            self.zero_grad()
            loss.backward()

            optimizer = optim.SGD(self.parameters(), lr=lr)
            # optimizer = optim.Adagrad(self.parameters(), lr=lr)
            # optimizer = optim.Adam(self.parameters(), lr=lr)
            optimizer.step()

        while abs(loss_prevprev - loss.item()) >= 0.01:

            out = self(input)

            loss_prevprev = loss_prev
            loss_prev = loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, target_cross)

            if lmbd is not None:
                loss += self.l2_reg(lmbd)

            if rho is not None:
                loss += self.Lip_reg(rho, parameters)
                
            if c is not None:
                self.LMT_reg(c)

            if np.mod(i, 5000) == 0:
                weights, biases = self.extract_weights()
                Lip = solve_SDP_multi.build_T_multi(weights, biases, net_dims)
                L = Lip["Lipschitz"]
                T = Lip["T"]
                L_W = 1
                for j in range(len(weights)):
                    L_W = L_W * np.linalg.norm(weights[j], 2)
                Lip_course.append(L)
                loss_course.append(loss.item())
                crossEntropyLoss = criterion(out, target_cross)
                CEloss_course.append(crossEntropyLoss)
                print('Train Epoch: {}; Loss: {:.6f}; CE-Loss: {:.6f}; Lipschitz: {:.3f}; Trivial Lipschitz: {:.3f}'.format(
                    i, loss.item(), crossEntropyLoss, L, L_W))
                # print(Lip["ok"])
            self.zero_grad()
            loss.backward()

            optimizer = optim.SGD(self.parameters(), lr=lr)
            # optimizer = optim.Adagrad(self.parameters(), lr=lr)
            # optimizer = optim.Adam(self.parameters(), lr=lr)
            optimizer.step()

            i += 1

        print('Train Epoch: {}; Loss: {:.6f}; CE-Loss: {:.6f}; Lipschitz: {:.3f}; Trivial Lipschitz: {:.3f}'.format(
                    i, loss.item(), crossEntropyLoss, L, L_W))

        if (lmbd is None) and (rho is None) and (c is None):
            Lip_dic = solve_SDP_multi.build_T_multi(weights, biases, net_dims)
            Lip_nom = Lip_dic["Lipschitz"]

        return Lip_course, loss_course, CEloss_course, T

    def train_Lipschitz(self, parameters, T):
        L_course_Lip = []
        loss_course_Lip = []
        CEloss_course_Lip = []
        Lip_prev = 0
        for i in range(5):
            print("Beginn ADMM Iteration # {:d}".format(i))
            print("Beginn Training")
            t1 = time.time()
            L_course, loss_course, CEloss_course, T = self.train(rho=rho, parameters=parameters) #loss update step
            timeLipTrain = time.time() - t1
            print("Training Complete after {} seconds.".format(timeLipTrain))
            for j in range(len(L_course)):
                L_course_Lip.append(L_course[j])
            for j in range(len(loss_course)):
                loss_course_Lip.append(loss_course[j])
            for j in range(len(CEloss_course)):
                CEloss_course_Lip.append(CEloss_course[j])
            weights, biases = self.extract_weights()
            print("Beginn parameter update step")
            t = time.time()
            for j in range(len(weights)):
                parameters.update({
                    'W{:d}'.format(j): matlab.double(np.array(weights, dtype=np.object)[j].tolist()),
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

        Lip_now = L_course[len(L_course)-1]
        while abs(Lip_prev - Lip_now) >= 1:
            i += 1
            Lip_prev = Lip_now
            print("Beginn ADMM Iteration # {:d}".format(i))
            print("Beginn Training")
            t1 = time.time()
            L_course, loss_course, CEloss_course, T = self.train(rho=rho, parameters=parameters) #loss update step
            timeLipTrain = time.time() - t1
            print("Training Complete after {} seconds.".format(timeLipTrain))
            for j in range(len(L_course)):
                L_course_Lip.append(L_course[j])
            for j in range(len(loss_course)):
                loss_course_Lip.append(loss_course[j])
            for j in range(len(CEloss_course)):
                CEloss_course_Lip.append(CEloss_course[j])
            weights, biases = self.extract_weights()
            print("Beginn parameter update step")
            t = time.time()
            for j in range(len(weights)):
                parameters.update({
                    'W{:d}'.format(j): matlab.double(np.array(weights, dtype=np.object)[j].tolist()),
                    })
            timeParameterUpdate = time.time() - t
            print("Parameter update step complete after {} seconds.".format(timeParameterUpdate))
            print("Beginn Solving SDP (Lipschitz and Y update step)")
            t = time.time()
            parameters = solve_SDP_multi.solve_SDP_multi(parameters, T, net_dims, rho, mu, ind_Lip, L_des) # Lipschitz and Y update steps
            timeSolveSDP = time.time() - t
            print("Solving SDP (Lipschitz and Y update step) complete after {} seconds.".format(timeSolveSDP))

            Lip_now = L_course[len(L_course)-1]
            timeADMM = time.time() - t1
            print("ADMM Iteration # {:d} complete after {} seconds.".format(i, timeADMM))

        return parameters, L_course_Lip, loss_course_Lip, CEloss_course_Lip


# Create Data
N = 50
np.random.seed(1612111977)
x = np.random.rand(N, 1)
y = np.random.rand(N, 1)

# Plot Data
plt.scatter(x, y)
plt.show()

# Create Input
input = torch.Tensor(np.concatenate((x, y), axis=1))

# Create Target
target = Variable(torch.zeros(input.size()))
for j in range(N):
    # print(x[j])
    if (x[j]-0.5)**2 + (y[j]-0.5)**2 <= 0.16:
        target[j, 1] += 1
    else:
        target[j, 0] += 1

# Create Target
target_cross = Variable(torch.zeros(N, dtype=torch.long))
for j in range(N):
    if (x[j]-0.5)**2 + (y[j]-0.5)**2 <= 0.04:
        target_cross[j] += 2
    elif (x[j]-0.5)**2 + (y[j]-0.5)**2 <= 0.16:
        target_cross[j] += 1
    else:
        target_cross[j] += 0


# Create NomNetz
netz = MeinNetz()
optimizer = optim.SGD(netz.parameters(), lr=lr)


print("Beginnning nominal NN training")
t = time.time()
Lip_course, loss_course, CEloss_course, T = netz.train()
timeNom = time.time() - t
print("Nominal Training Complete after {} seconds".format(timeNom))

weights, biases = netz.extract_weights()
Lip = solve_SDP_multi.build_T_multi(weights, biases, net_dims)

Lip_dic = solve_SDP_multi.build_T_multi(weights, biases, net_dims)
Lip_nom = Lip_dic["Lipschitz"]

torch.save(netz, '2D_NomModel.pt')

# plot losscourse
plt.plot(loss_course)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Loss Nom')
plt.show()

# plot CElosscourse
plt.plot(CEloss_course)
plt.xlabel('# 10^4 iterations')
plt.ylabel('CE-Loss Nom')
plt.show()

# plot Lip_course
plt.plot(Lip_course)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Lip_course Nom')
plt.show()

# plot Lip_course with time
plt.plot(Lip_course)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Lip_course Nom')
plt.title('TimeNom = '+str(timeNom))
plt.show()

# get output
out = F.softmax(netz(input))
# print(out)

# scatter
x0 = []
x1 = []
x2 = []
y0 = []
y1 = []
y2 = []
for i in range(len(x)):
    if np.argmax(out.detach().numpy()[i, :]) == 0:
        x0 = np.append(x0, x[i])
        y0 = np.append(y0, y[i])
    elif np.argmax(out.detach().numpy()[i, :]) == 1:
        x1 = np.append(x1, x[i])
        y1 = np.append(y1, y[i])
    if np.argmax(out.detach().numpy()[i, :]) == 2:
        x2 = np.append(x2, x[i])
        y2 = np.append(y2, y[i])

plt.scatter(x0, y0)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.title('predicted classification Nom')
plt.show()

# true classification
x0_true = []
x1_true = []
x2_true = []
y0_true = []
y1_true = []
y2_true = []
for i in range(len(x)):
    if target_cross[i].item() == 0:
        x0_true = np.append(x0_true, x[i])
        y0_true = np.append(y0_true, y[i])
    elif target_cross[i].item() == 1:
        x1_true = np.append(x1_true, x[i])
        y1_true = np.append(y1_true, y[i])
    if target_cross[i].item() == 2:
        x2_true = np.append(x2_true, x[i])
        y2_true = np.append(y2_true, y[i])

circle_green = plt.Circle((0.5, 0.5), 0.2, color='g', fill=False)
circle_orange = plt.Circle((0.5, 0.5), 0.4, color='r', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle_green)
ax.add_artist(circle_orange)

plt.scatter(x0_true, y0_true)
plt.scatter(x1_true, y1_true)
plt.scatter(x2_true, y2_true)
plt.title('true classification')
plt.show()


# cut
a = np.linspace(0, 1, num=100)
b = 0.5*np.ones(100)
ybottom = np.zeros(N)
input_cut = np.random.rand(100, 2)
for i in range(len(a)):
    input_cut[i] = np.append(a[i], b[i])
input_cut = torch.Tensor(input_cut)

out_cut = F.softmax(netz(input_cut))

plt.plot(a, out_cut.detach().numpy()[:, 0])
plt.plot(a, out_cut.detach().numpy()[:, 1])
plt.plot(a, out_cut.detach().numpy()[:, 2])
plt.title('out_cut Nom')
plt.show()

# area
numArea = 5000
input_area = np.random.rand(numArea, 2)
input_area = torch.Tensor(input_area)
out_area = F.softmax(netz(input_area))
x0 = []
x1 = []
x2 = []
y0 = []
y1 = []
y2 = []
for i in range(len(input_area)):
    if np.argmax(out_area.detach().numpy()[i, :]) == 0:
        x0 = np.append(x0, input_area[i, 0].item())
        y0 = np.append(y0, input_area[i, 1].item())
    elif np.argmax(out_area.detach().numpy()[i, :]) == 1:
        x1 = np.append(x1, input_area[i, 0].item())
        y1 = np.append(y1, input_area[i, 1].item())
    if np.argmax(out_area.detach().numpy()[i, :]) == 2:
        x2 = np.append(x2, input_area[i, 0].item())
        y2 = np.append(y2, input_area[i, 1].item())

counter = 0
for i in range(len(input_area)):
    if (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 <= 0.04) and (np.argmax(out_area.detach().numpy()[i, :]) == 2)) or (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 >= 0.04) and ((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 <= 0.16) and (np.argmax(out_area.detach().numpy()[i, :]) == 1)) or (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 >= 0.16) and (np.argmax(out_area.detach().numpy()[i, :]) == 0)):
        counter += 1
accuracy = counter*100/numArea
print("Test Accuracy: {}".format(accuracy))

circle_green = plt.Circle((0.5, 0.5), 0.2, color='g', fill=False)
circle_orange = plt.Circle((0.5, 0.5), 0.4, color='r', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle_green)
ax.add_artist(circle_orange)

plt.scatter(x0, y0)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x, y, c='k')
plt.title('out_area Nom')
plt.show()


# NN with L2 regularizer
net_L2 = MeinNetz()

print("Beginnning L2 training")
t = time.time()
Lip_course_L2, loss_course_L2, CEloss_course_L2, T = net_L2.train(lmbd=lmbd)
timeL2 = time.time() - t
print("L2 Training Complete after {} seconds".format(timeL2))

weights_L2, biases_L2 = net_L2.extract_weights()
Lip_L2 = solve_SDP_multi.build_T_multi(weights_L2, biases_L2, net_dims)

torch.save(net_L2, '2D_L2Model.pt')

# plot losscourse
plt.plot(loss_course_L2)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Loss L2')
plt.show()

# plot CElosscourse
plt.plot(CEloss_course_L2)
plt.xlabel('# 10^4 iterations')
plt.ylabel('CE-Loss L2')
plt.show()

# plot Lip_course
plt.plot(Lip_course_L2)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Lip_course L2')
plt.show()

# plot Lip_course with time
plt.plot(Lip_course)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Lip_course Nom')
plt.title('TimeL2 = '+str(timeL2))
plt.show()

# get output
out_L2 = F.softmax(net_L2(input))
# print(out_L2)

# scatter
x0 = []
x1 = []
x2 = []
y0 = []
y1 = []
y2 = []
for i in range(N):
    if np.argmax(out_L2.detach().numpy()[i, :]) == 0:
        x0 = np.append(x0, x[i])
        y0 = np.append(y0, y[i])
    elif np.argmax(out_L2.detach().numpy()[i, :]) == 1:
        x1 = np.append(x1, x[i])
        y1 = np.append(y1, y[i])
    if np.argmax(out_L2.detach().numpy()[i, :]) == 2:
        x2 = np.append(x2, x[i])
        y2 = np.append(y2, y[i])

plt.scatter(x0, y0)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.title('predicted classification L2')
plt.show()

# true classification
circle_green = plt.Circle((0.5, 0.5), 0.2, color='g', fill=False)
circle_orange = plt.Circle((0.5, 0.5), 0.4, color='r', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle_green)
ax.add_artist(circle_orange)

plt.scatter(x0_true, y0_true)
plt.scatter(x1_true, y1_true)
plt.scatter(x2_true, y2_true)
plt.title('true classification')
plt.show()


# cut
out_cut_L2 = F.softmax(net_L2(input_cut))

plt.plot(a, out_cut_L2.detach().numpy()[:, 0])
plt.plot(a, out_cut_L2.detach().numpy()[:, 1])
plt.plot(a, out_cut_L2.detach().numpy()[:, 2])
plt.title('out_cut L2')
plt.show()

# area
out_area_L2 = F.softmax(net_L2(input_area))
x0 = []
x1 = []
x2 = []
y0 = []
y1 = []
y2 = []
for i in range(len(input_area)):
    if np.argmax(out_area_L2.detach().numpy()[i, :]) == 0:
        x0 = np.append(x0, input_area[i, 0].item())
        y0 = np.append(y0, input_area[i, 1].item())
    elif np.argmax(out_area_L2.detach().numpy()[i, :]) == 1:
        x1 = np.append(x1, input_area[i, 0].item())
        y1 = np.append(y1, input_area[i, 1].item())
    if np.argmax(out_area_L2.detach().numpy()[i, :]) == 2:
        x2 = np.append(x2, input_area[i, 0].item())
        y2 = np.append(y2, input_area[i, 1].item())

counter = 0
for i in range(len(input_area)):
    if (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 <= 0.04) and (np.argmax(out_area_L2.detach().numpy()[i, :]) == 2)) or (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 >= 0.04) and ((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 <= 0.16) and (np.argmax(out_area_L2.detach().numpy()[i, :]) == 1)) or (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 >= 0.16) and (np.argmax(out_area_L2.detach().numpy()[i, :]) == 0)):
        counter += 1
accuracy = counter*100/numArea
print("Test Accuracy: {}".format(accuracy))

circle_green = plt.Circle((0.5, 0.5), 0.2, color='g', fill=False)
circle_orange = plt.Circle((0.5, 0.5), 0.4, color='r', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle_green)
ax.add_artist(circle_orange)

plt.scatter(x0, y0)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x, y, c='k')
plt.title('out_area L2')
plt.show()


# NN with Lipschitz regularizer
net_Lip = MeinNetz()
net_Lip.load_state_dict(net_L2.state_dict())

L_des = Lip_L2["Lipschitz"]

print("Beginnning parameters = solve_SDP1")
t1 = time.time()
parameters = solve_SDP_multi.initialize_parameters(weights, biases)
timeSolveSDP1 = time.time() - t1
print("Complete parameters = solve_SDP1 after {} seconds".format(timeSolveSDP1))
print("Beginnning parameters = solve_SDP2")
t2 = time.time()
parameters_L2 = solve_SDP_multi.initialize_parameters(weights_L2, biases_L2)
timeSolveSDP2 = time.time() - t2
print("Complete parameters = solve_SDP2 after {} seconds".format(timeSolveSDP2))

init = 1  # 1 initialize from L2-NN, 2 initialize from nominal NN
if init == 1:
    T = Lip_L2["T"]
    print("Beginnning LipSDP training")
    t3 = time.time()
    parameters_Lip, Lip_course_Lip, loss_course_Lip, CEloss_course_Lip = net_Lip.train_Lipschitz(parameters=parameters_L2, T=Lip_L2["T"])
    timeTrainSDP = time.time() - t3
    print("LipSDP training complete after {} seconds".format(timeTrainSDP))
else:
    T = Lip_dic["T"]
    print("Beginnning LipSDP training")
    t3 = time.time()
    parameters_Lip, Lip_course_Lip, loss_course_Lip, CEloss_course_Lip = net_Lip.train_Lipschitz(parameters=parameters, T=Lip_dic["T"])   
    timeTrainSDP = time.time() - t3
    print("LipSDP training complete after {} seconds".format(timeTrainSDP))

timeFullSDP = time.time() - t1
print("Full LipSDP training complete after {} seconds".format(timeFullSDP))

net_Lip2 = type(net_Lip)()
net_Lip2.load_state_dict(net_Lip.state_dict())
with torch.no_grad():
    net_Lip2.lin1.weight = torch.nn.Parameter(torch.tensor(parameters_Lip['W0_bar']))
    net_Lip2.lin2.weight = torch.nn.Parameter(torch.tensor(parameters_Lip['W1_bar']))

weights_Lip, biases_Lip = net_Lip.extract_weights()
weights_Lip2, biases_Lip2 = net_Lip2.extract_weights()

Lip_Lip = solve_SDP_multi.build_T_multi(weights_Lip, biases_Lip, net_dims)
Lip_Lip2 = solve_SDP_multi.build_T_multi(weights_Lip2, biases_Lip2, net_dims)

torch.save(net_Lip, '2D_LipModel.pt')

# plot losscourse
plt.plot(loss_course_Lip)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Loss Lip')
plt.show()

# plot losscourse
plt.plot(CEloss_course_Lip)
plt.xlabel('# 10^4 iterations')
plt.ylabel('CE-Loss Lip')
plt.show()

# plot Lip_course
plt.plot(Lip_course_Lip)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Lip_course Lip')
plt.show()

# plot Lip_course with time
plt.plot(Lip_course)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Lip_course Nom')
plt.title('TimeSolveSDP1 = '+str(timeSolveSDP1)+', TimeSolveSDP2 = '+str(timeSolveSDP2)+', TimeTrainSDP = '+str(timeTrainSDP)+', TimeFullSDP = '+str(timeFullSDP))
plt.show()

# get output
out_Lip = F.softmax(net_Lip(input))
# print(out_Lip)

# scatter
x0 = []
x1 = []
x2 = []
y0 = []
y1 = []
y2 = []
for i in range(N):
    if np.argmax(out_Lip.detach().numpy()[i, :]) == 0:
        x0 = np.append(x0, x[i])
        y0 = np.append(y0, y[i])
    elif np.argmax(out_Lip.detach().numpy()[i, :]) == 1:
        x1 = np.append(x1, x[i])
        y1 = np.append(y1, y[i])
    if np.argmax(out_Lip.detach().numpy()[i, :]) == 2:
        x2 = np.append(x2, x[i])
        y2 = np.append(y2, y[i])

plt.scatter(x0, y0)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.title('predicted classification Lip')
plt.show()

# true classification
circle_green = plt.Circle((0.5, 0.5), 0.2, color='g', fill=False)
circle_orange = plt.Circle((0.5, 0.5), 0.4, color='r', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle_green)
ax.add_artist(circle_orange)

plt.scatter(x0_true, y0_true)
plt.scatter(x1_true, y1_true)
plt.scatter(x2_true, y2_true)
plt.title('true classification')
plt.show()


# cut
out_cut_Lip = F.softmax(net_Lip(input_cut))

plt.plot(a, out_cut_Lip.detach().numpy()[:, 0])
plt.plot(a, out_cut_Lip.detach().numpy()[:, 1])
plt.plot(a, out_cut_Lip.detach().numpy()[:, 2])
plt.title('out_cut Lip')
plt.show()

# area
out_area_Lip = F.softmax(net_Lip(input_area))
x0 = []
x1 = []
x2 = []
y0 = []
y1 = []
y2 = []
for i in range(len(input_area)):
    if np.argmax(out_area_Lip.detach().numpy()[i, :]) == 0:
        x0 = np.append(x0, input_area[i, 0].item())
        y0 = np.append(y0, input_area[i, 1].item())
    elif np.argmax(out_area_Lip.detach().numpy()[i, :]) == 1:
        x1 = np.append(x1, input_area[i, 0].item())
        y1 = np.append(y1, input_area[i, 1].item())
    if np.argmax(out_area_Lip.detach().numpy()[i, :]) == 2:
        x2 = np.append(x2, input_area[i, 0].item())
        y2 = np.append(y2, input_area[i, 1].item())

counter = 0
for i in range(len(input_area)):
    if (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 <= 0.04) and (np.argmax(out_area_Lip.detach().numpy()[i, :]) == 2)) or (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 >= 0.04) and ((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 <= 0.16) and (np.argmax(out_area_Lip.detach().numpy()[i, :]) == 1)) or (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 >= 0.16) and (np.argmax(out_area_Lip.detach().numpy()[i, :]) == 0)):
        counter += 1
accuracy = counter*100/numArea
print("Test Accuracy: {}".format(accuracy))

circle_green = plt.Circle((0.5, 0.5), 0.2, color='g', fill=False)
circle_orange = plt.Circle((0.5, 0.5), 0.4, color='r', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle_green)
ax.add_artist(circle_orange)

plt.scatter(x0, y0)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x, y, c='k')
plt.title('out_area Lip')
plt.show()


# NN with LMT
net_LMT = MeinNetz()
net_LMT.load_state_dict(net_L2.state_dict())

print("Beginnning LMT training")
t = time.time()
Lip_course_LMT, loss_course_LMT, CEloss_course_LMT, T = net_LMT.train(c=c)
timeLMT = time.time() - t
print("Training Complete after {} seconds".format(timeLMT))

weights_LMT, biases_LMT = net_LMT.extract_weights()
Lip_LMT = solve_SDP_multi.build_T_multi(weights_LMT, biases_LMT, net_dims)

torch.save(net_LMT, '2D_LMTModel.pt')

# plot losscourse
plt.plot(loss_course_LMT)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Loss LMT')
plt.show()

# plot CElosscourse
plt.plot(CEloss_course_LMT)
plt.xlabel('# 10^4 iterations')
plt.ylabel('CE-Loss LMT')
plt.show()

# plot Lip_course
plt.plot(Lip_course_LMT)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Lip_course LMT')
plt.show()

# plot Lip_course with time
plt.plot(Lip_course)
plt.xlabel('# 10^4 iterations')
plt.ylabel('Lip_course Nom')
plt.title('TimeLMT = '+str(timeLMT))
plt.show()

# get output
out_LMT = F.softmax(net_LMT(input))
# print(out_LMT)

# scatter
x0 = []
x1 = []
x2 = []
y0 = []
y1 = []
y2 = []
for i in range(N):
    if np.argmax(out_LMT.detach().numpy()[i, :]) == 0:
        x0 = np.append(x0, x[i])
        y0 = np.append(y0, y[i])
    elif np.argmax(out_LMT.detach().numpy()[i, :]) == 1:
        x1 = np.append(x1, x[i])
        y1 = np.append(y1, y[i])
    if np.argmax(out_LMT.detach().numpy()[i, :]) == 2:
        x2 = np.append(x2, x[i])
        y2 = np.append(y2, y[i])

plt.scatter(x0, y0)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.title('predicted classification LMT')
plt.show()

# true classification
circle_green = plt.Circle((0.5, 0.5), 0.2, color='g', fill=False)
circle_orange = plt.Circle((0.5, 0.5), 0.4, color='r', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle_green)
ax.add_artist(circle_orange)

plt.scatter(x0_true, y0_true)
plt.scatter(x1_true, y1_true)
plt.scatter(x2_true, y2_true)
plt.title('true classification')
plt.show()


# cut
out_cut_LMT = F.softmax(net_LMT(input_cut))

plt.plot(a, out_cut_LMT.detach().numpy()[:, 0])
plt.plot(a, out_cut_LMT.detach().numpy()[:, 1])
plt.plot(a, out_cut_LMT.detach().numpy()[:, 2])
plt.title('out_cut LMT')
plt.show()

# area
out_area_LMT = F.softmax(net_LMT(input_area))
x0 = []
x1 = []
x2 = []
y0 = []
y1 = []
y2 = []
for i in range(len(input_area)):
    if np.argmax(out_area_LMT.detach().numpy()[i, :]) == 0:
        x0 = np.append(x0, input_area[i, 0].item())
        y0 = np.append(y0, input_area[i, 1].item())
    elif np.argmax(out_area_LMT.detach().numpy()[i, :]) == 1:
        x1 = np.append(x1, input_area[i, 0].item())
        y1 = np.append(y1, input_area[i, 1].item())
    if np.argmax(out_area_LMT.detach().numpy()[i, :]) == 2:
        x2 = np.append(x2, input_area[i, 0].item())
        y2 = np.append(y2, input_area[i, 1].item())

counter = 0
for i in range(len(input_area)):
    if (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 <= 0.04) and (np.argmax(out_area_LMT.detach().numpy()[i, :]) == 2)) or (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 >= 0.04) and ((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 <= 0.16) and (np.argmax(out_area_LMT.detach().numpy()[i, :]) == 1)) or (((input_area[i, 0]-0.5)**2 + (input_area[i, 1]-0.5)**2 >= 0.16) and (np.argmax(out_area_LMT.detach().numpy()[i, :]) == 0)):
        counter += 1
accuracy = counter*100/numArea
print("Test Accuracy: {}".format(accuracy))

circle_green = plt.Circle((0.5, 0.5), 0.2, color='g', fill=False)
circle_orange = plt.Circle((0.5, 0.5), 0.4, color='r', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle_green)
ax.add_artist(circle_orange)

plt.scatter(x0, y0)
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x, y, c='k')
plt.title('out_area LMT')
plt.show()


# plt.plot(loss_course)
# plt.xlabel('# iterations')
# plt.ylabel('loss Loss')
# plt.show()

# out = netz(input)
# print(out)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(x, y, target[:, 1], cmap='Greens')
# ax.scatter3D(x, y, out.detach().numpy()[:, 1], cmap='Greens')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('class')
# ax.set_title('2D classifier')
# ax.legend(['Targets', 'Nom'])


# Predictions
out = F.softmax(netz(input))
out_L2 = F.softmax(net_L2(input))
out_Lip = F.softmax(net_Lip(input))
out_Lip2 = F.softmax(net_Lip2(input))
out_LMT = F.softmax(net_LMT(input))

# Plots
now = datetime.now()
date = now.strftime("%Y-%m-%d_%H-%M-%S")

# plt.plot(input, target)
# plt.plot(input, out.detach().numpy())
# plt.plot(input, out_L2.detach().numpy())
# plt.plot(input, out_Lip.detach().numpy())
# plt.plot(input, out_Lip2.detach().numpy())
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Targets', 'Nom', 'L2', 'Lip', 'Lip2'])
# plt.savefig('Results/'+date+'.png')
# plt.show()

plt.plot(loss_course, label='Nom')
plt.plot(loss_course_L2, label='L2')
plt.plot(np.array(loss_course_Lip).reshape(np.array(loss_course_Lip).size, 1), label='LipSDP')
plt.plot(loss_course_LMT, label='LMT')
plt.xlabel('# 10^4 iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Results/'+str(mu)+str(rho)+date+'_loss.png')
plt.show()

plt.plot(CEloss_course, label='Nom')
plt.plot(CEloss_course_L2, label='L2')
plt.plot(np.array(CEloss_course_Lip).reshape(np.array(CEloss_course_Lip).size, 1), label='LipSDP')
plt.plot(CEloss_course_LMT, label='LMT')
plt.xlabel('# 10^4 iterations')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.savefig('Results/'+str(mu)+str(rho)+date+'_CEloss.png')
plt.show()

plt.plot(Lip_course, label='Nom')
plt.plot(Lip_course_L2, label='L2')
plt.plot(np.array(Lip_course_Lip).reshape(np.array(Lip_course_Lip).size, 1), label='LipSDP')
plt.plot(Lip_course_LMT, label='LMT')
plt.xlabel('# 10^4 iterations')
plt.ylabel('Lipschitz bound')
plt.legend()
plt.savefig('Results/'+str(mu)+str(rho)+date+'_Lip.png')
plt.show()


# cut
plt.plot(a, out_cut.detach().numpy()[:, 0], '-b', label='Nom0')
plt.plot(a, out_cut.detach().numpy()[:, 1], '-r', label='Nom1')
plt.plot(a, out_cut.detach().numpy()[:, 2], '-g', label='Nom2')
plt.plot(a, out_cut_L2.detach().numpy()[:, 0], '--b', label='L20')
plt.plot(a, out_cut_L2.detach().numpy()[:, 1], '--r', label='L21')
plt.plot(a, out_cut_L2.detach().numpy()[:, 2], '--g', label='L22')
plt.plot(a, out_cut_Lip.detach().numpy()[:, 0], ':b', label='Lip0')
plt.plot(a, out_cut_Lip.detach().numpy()[:, 1], ':r', label='Lip1')
plt.plot(a, out_cut_Lip.detach().numpy()[:, 2], ':g', label='Lip2')
# plt.plot(a, out_cut_LMT.detach().numpy()[:, 0], '-.b')
# plt.plot(a, out_cut_LMT.detach().numpy()[:, 1], '-.r')
# plt.plot(a, out_cut_LMT.detach().numpy()[:, 2], '-.g')
legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
plt.title('out_cut')
plt.show()


# Save Hyperparameters
hyper = {
    'lr': lr,
    'n_x': net_dims[0],
    'n_h': net_dims[1],
    'n_y': net_dims[2],
    'rho': rho,
    'mu': mu,
    'lambda': lmbd,
    'c': c,
    'ind_Lip': ind_Lip,
    'Lip_des': L_des
}

# Save data and results
data = {
    'input': np.array(input, dtype=np.float64),
    'target': np.array(target, dtype=np.float64),
    'out': np.array(out.detach().numpy(), dtype=np.float64),
    'out_L2': np.array(out_L2.detach().numpy(), dtype=np.float64),
    'out_Lip': np.array(out_Lip.detach().numpy(), dtype=np.float64),
    'out_Lip2': np.array(out_Lip2.detach().numpy(), dtype=np.float64),
    'Lip': Lip["Lipschitz"],
    'Lip_L2': Lip_L2["Lipschitz"],
    'Lip_Lip': Lip_Lip["Lipschitz"],
    'Lip_Lip2': Lip_Lip2["Lipschitz"],
    # 'MSELoss': netz.evaluate_MSELoss(input, target).detach().numpy(),
    # 'MSELoss_L2': net_L2.evaluate_MSELoss(input, target).detach().numpy(),
    # 'MSELoss_Lip': net_Lip.evaluate_MSELoss(input, target).detach().numpy(),
    # 'MSELoss_Lip2': net_Lip2.evaluate_MSELoss(input, target).detach().numpy(),
    # 'MSELoss_test': net.evaluate_MSELoss(input,target_test).detach().numpy(),
    # 'MSELoss_test_L2': net_L2.evaluate_MSELoss(input,target_test).detach().numpy(),
    # 'MSELoss_test_Lip': net_Lip.evaluate_MSELoss(input,target_test).detach().numpy(),
    # 'MSELoss_test_Lip2': net_Lip2.evaluate_MSELoss(input,target_test).detach().numpy(), 
    'L_course': np.array(Lip_course),
    'L_course_L2': np.array(Lip_course_L2),
    'L_course_Lip': np.array(np.array(Lip_course_Lip).reshape(np.array(Lip_course_Lip).size,1)),
    'MSE_course': np.array(loss_course),
    'MSE_course_L2': np.array(loss_course_L2),
    'MSE_course_Lip': np.array(np.array(loss_course_Lip).reshape(np.array(loss_course_Lip).size,1))       
    }

res = {}
res["hyper"] = hyper
res["data"] = data
res["weights"] = weights
res["weights_L2"] = weights_L2
res["weights_Lip"] = weights_Lip
res["weights_Lip2"] = weights_Lip2
res["biases"] = biases
res["biases_L2"] = biases_L2
res["biases_Lip"] = biases_Lip
res["biases_Lip2"] = biases_Lip2

fname = os.path.join(os.getcwd(), 'Results/res_'+date+'.mat')
data = {'res': np.array(res)}
savemat(fname, data)
