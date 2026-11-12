# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 02:15:59 2020

@author: paul_
"""


import torch
import matlab.engine
import numpy as np
import json
import atexit
from experiment_paths import REPO_ROOT, run_dir, output_path


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        engine = matlab.engine.start_matlab()
        try:
            engine.addpath(str(REPO_ROOT), nargout=0)
            engine.cd(str(run_dir()), nargout=0)
            engine.configure_solver_paths(nargout=0)
        except Exception:
            engine.quit()
            raise
        _engine = engine
        atexit.register(engine.quit)
    return _engine


def build_T_multi(weights, biases, net_dims):
    x = matlab.int64([net_dims])

    parameters = {}
    weights_dict = {}
    biases_dict = {}
    for i in range(len(weights)):
        parameters.update({
            'W{:d}'.format(i): matlab.double(np.array(weights, dtype=object)[i].tolist()),
            })
        parameters.update({
            'b{:d}'.format(i): matlab.double(np.array(biases, dtype=object)[i].tolist()),
            })
        weights_dict.update({
            'W{:d}'.format(i): np.array(weights, dtype=object)[i].tolist() })
        biases_dict.update({
            'b{:d}'.format(i): np.array(biases, dtype=object)[i].tolist() })

    with open(output_path('weights.json'), 'w') as f:  # writing JSON object
        json.dump(weights_dict, f)
    with open(output_path('biases.json'), 'w') as f:  # writing JSON object
        json.dump(biases_dict, f)
    Lip = get_engine().calculate_Lipschitz_multi(parameters, x)
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

    parameters = get_engine().solve_sdp_multi(parameters, x, sdp)

    return parameters


def initialize_parameters(weights, biases):
    parameters = {}
    for i in range(len(weights)):
        parameters.update({
            'W{:d}'.format(i): matlab.double(np.array(weights, dtype=object)[i].tolist()),
            })
        parameters.update({
            'W{:d}_bar'.format(i): matlab.double(np.array(weights, dtype=object)[i].tolist()),
            })
        parameters.update({
            'b{:d}'.format(i): matlab.double(np.array(biases, dtype=object)[i].tolist()),
            })
        parameters.update({
            'Y{:d}'.format(i): matlab.double(torch.zeros(weights[i].shape).tolist()),
            })
        parameters.update({
            'Yb{:d}'.format(i): matlab.double(torch.zeros(biases[i].shape).tolist()),
            })
    return parameters

