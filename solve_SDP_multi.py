# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 02:15:59 2020

@author: paul_
"""


import torch
import matlab.engine
import numpy as np
import json


eng = matlab.engine.start_matlab()


def build_T_multi(weights, biases, net_dims):
    x = matlab.int64([net_dims])

    parameters = {}
    weights_dict = {}
    biases_dict = {}
    for i in range(len(weights)):
        parameters.update({
            'W{:d}'.format(i): matlab.double(np.array(weights, dtype=np.object)[i].tolist()),
            })
        parameters.update({
            'b{:d}'.format(i): matlab.double(np.array(biases, dtype=np.object)[i].tolist()),
            })
        weights_dict.update({
            'W{:d}'.format(i): np.array(weights, dtype=np.object)[i].tolist() })
        biases_dict.update({
            'b{:d}'.format(i): np.array(biases, dtype=np.object)[i].tolist() })

    with open('weights.json', 'w') as f:  # writing JSON object
        json.dump(weights_dict, f)
    with open('biases.json', 'w') as f:  # writing JSON object
        json.dump(biases_dict, f)
    Lip = eng.calculate_Lipschitz_multi(parameters, x)
    return Lip


def solve_SDP_multi(parameters, T, net_dims, rho, mu, ind_Lip, L_des):
    x = matlab.int64([net_dims])
    sdp = {
        'T': matlab.double([T]),
        'rho': matlab.double([rho]),
        'mu': matlab.double([mu]),
        'ind_Lip': matlab.int64([ind_Lip]),
        'L_des': matlab.double([L_des])
    }

    parameters = eng.solve_sdp_multi(parameters, x, sdp)

    return parameters


def initialize_parameters(weights, biases):
    parameters = {}
    for i in range(len(weights)):
        parameters.update({
            'W{:d}'.format(i): matlab.double(np.array(weights, dtype=np.object)[i].tolist()),
            })
        parameters.update({
            'W{:d}_bar'.format(i): matlab.double(np.array(weights, dtype=np.object)[i].tolist()),
            })
        parameters.update({
            'b{:d}'.format(i): matlab.double(np.array(biases, dtype=np.object)[i].tolist()),
            })
        parameters.update({
            'Y{:d}'.format(i): matlab.double(torch.zeros(weights[i].shape).tolist()),
            })
        parameters.update({
            'Yb{:d}'.format(i): matlab.double(torch.zeros(biases[i].shape).tolist()),
            })
    return parameters
