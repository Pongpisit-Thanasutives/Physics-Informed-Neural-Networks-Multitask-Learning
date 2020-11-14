#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../')

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# from FBSNNs import FBSNN
from MTL_FBSNNs_Allen100D import FBSNN


class AllenCahn(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, layers, mode, activation)

    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return - Y + Y ** 3  # M x 1

    def g_tf(self, X):
        return 1.0 / (2.0 + 0.4 * torch.sum(X ** 2, 1, keepdim=True))
    
    def aux_g_tf(self, X):
        return 1.0 / (2.0 + 0.3 * torch.sum(X ** 2, 1, keepdim=True))

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z)  # M x D

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        return super().sigma_tf(t, X, Y)  # M x D x D

M = 100  # number of trajectories (batch size)
N = 50  # number of time snapshots
D = 100  # number of dimensions

layers = [D + 1] + 4 * [256] + [1]

Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
T = 1.0

"Available architectures"
mode = "FC"  # FC, Resnet and NAIS-Net are available
activation = "Sine"  # sine and ReLU are available
model = AllenCahn(Xi, T, M, N, D, layers, mode, activation)


weights_path = 'allen_uncert.pth'
model.model.load_state_dict(torch.load(weights_path))


def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    r = 0.05
    sigma_max = 0.4
    return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1


t_test, W_test = model.fetch_minibatch()
X_pred, Y_pred = model.predict(Xi, t_test, W_test)

if type(t_test).__module__ != 'numpy':
    t_test = t_test.cpu().numpy()
if type(X_pred).__module__ != 'numpy':
    X_pred = X_pred.cpu().detach().numpy()
if type(Y_pred).__module__ != 'numpy':
    Y_pred = Y_pred.cpu().detach().numpy()

Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                    [M, -1, 1])


errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
mean_errors = np.mean(errors, 0)
std_errors = np.std(errors, 0)

print('averaged mean errors across time', np.mean(mean_errors))
print('averaged std errors across time', np.mean(std_errors))