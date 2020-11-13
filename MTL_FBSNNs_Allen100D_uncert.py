# My stuff here
import ct_module

import numpy as np
from abc import ABC, abstractmethod
import time

import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self, layers, length):
        super(MyModel, self).__init__()
        
        self.length = length
        # Task 1 network
        self.start1 = nn.Linear(layers[0], layers[1])
        self.processes1 = nn.ModuleList([nn.Linear(layers[1], layers[1]) for i in range(self.length)])
        self.end1 = nn.Linear(layers[-2], 1)
        
        # Task 2 network
        self.start2 = nn.Linear(layers[0], layers[1])
        self.processes2 = nn.ModuleList([nn.Linear(layers[1], layers[1]) for i in range(self.length)])
        self.end2 = nn.Linear(layers[-2], 1)
        
        # cross_stiches
        self.cross_stiches = nn.ModuleList([ct_module.CrossStich() for i in range(self.length+1)])
        
        # init weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.0)
                
    def propagate(self, x1, x2):
        feature1, feature2 = torch.tanh(self.start1(x1)), torch.tanh(self.start2(x2))
        feature1, feature2 = self.cross_stiches[0](feature1, feature2)
        
        for i in range(self.length):
            feature1, feature2 = torch.tanh(self.processes1[i](feature1)), torch.tanh(self.processes2[i](feature2))
            feature1, feature2 = self.cross_stiches[i+1](feature1, feature2)
            
        return self.end1(feature1), self.end2(feature2)
    
    def forward(self, x1, x2):
        return self.propagate(x1, x2)

class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):

        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True

        else:
            self.device = torch.device("cpu")

        #  We set a random seed to ensure that your results are reproducible
        # torch.manual_seed(0)

        self.Xi = torch.from_numpy(Xi).float().to(self.device)  # initial point
        self.Xi.requires_grad = True

        self.T = T  # terminal time
        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        
        # Model
        self.model = MyModel(layers, 3).cuda()
        
        # Only for uncert
        self.log_vars = nn.Parameter(torch.zeros((2)))

        # Record the loss
        self.training_loss = []
        self.iteration = []

    def net_u(self, t, X_t1, X_t2):  # M x 1, M x D
        input1 = torch.cat((t, X_t1), 1)
        input2 = torch.cat((t, X_t2), 1)
        
        u1, u2 = self.model(input1, input2)  # M x 1
        Du1 = torch.autograd.grad(outputs=[u1], inputs=[X_t1], grad_outputs=torch.ones_like(u1), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]
        Du2 = torch.autograd.grad(outputs=[u2], inputs=[X_t2], grad_outputs=torch.ones_like(u2), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]
        return [u1, Du1], [u2, Du2]

    def Dg_tf(self, X):  # M x D
        g = self.g_tf(X)
        Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]  # M x D
        return Dg
    
    def aux_Dg_tf(self, X):  # M x D
        g = self.aux_g_tf(X)
        Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]  # M x D
        return Dg

    def loss_function(self, t, W, Xi):
        loss1 = 0; loss2 = 0; total_loss = 0
        X_list = []
        Y_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]

        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)  # M x D
        X0_t1 = X0
        X0_t2 = X0
        [Y0_t1, Z0_t1], [Y0_t2, Z0_t2] = self.net_u(t0, X0_t1, X0_t2)  # M x 1, M x D

        X_list.append(X0_t1)
        Y_list.append(Y0_t1)

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            
            X1_t1 = X0_t1 + self.mu_tf(t0, X0_t1, Y0_t1, Z0_t1) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0_t1, Y0_t1), (W1 - W0).unsqueeze(-1)), dim=-1)
            X1_t2 = X0_t2 + self.mu_tf(t0, X0_t2, Y0_t2, Z0_t2) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0_t2, Y0_t2), (W1 - W0).unsqueeze(-1)), dim=-1)
            
            Y1_tilde_t1 = Y0_t1 + self.phi_tf(t0, X0_t1, Y0_t1, Z0_t1) * (t1 - t0) + torch.sum(
                Z0_t1 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0_t1, Y0_t1), (W1 - W0).unsqueeze(-1))), dim=1,
                keepdim=True)
            Y1_tilde_t2 = Y0_t2 + self.phi_tf(t0, X0_t2, Y0_t2, Z0_t2) * (t1 - t0) + torch.sum(
                Z0_t2 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0_t2, Y0_t2), (W1 - W0).unsqueeze(-1))), dim=1,
                keepdim=True)
            
            [Y1_t1, Z1_t1], [Y1_t2, Z1_t2] = self.net_u(t1, X1_t1, X1_t2)

            loss1 += torch.sum(torch.pow(Y1_t1 - Y1_tilde_t1, 2))
            loss2 += torch.sum(torch.pow(Y1_t2 - Y1_tilde_t2, 2))

            t0 = t1
            W0 = W1
            X0_t1 = X1_t1
            Y0_t1 = Y1_t1
            Z0_t1 = Z1_t1
            X0_t2 = X1_t2
            Y0_t2 = Y1_t2
            Z0_t2 = Z1_t2

            X_list.append(X0_t1)
            Y_list.append(Y0_t1)

        loss1 += torch.sum(torch.pow(Y1_t1 - self.g_tf(X1_t1), 2))
        loss1 += torch.sum(torch.pow(Z1_t1 - self.Dg_tf(X1_t1), 2))
        loss2 += torch.sum(torch.pow(Y1_t2 - self.aux_g_tf(X1_t2), 2))
        loss2 += torch.sum(torch.pow(Z1_t2 - self.aux_Dg_tf(X1_t2), 2))

        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        loss1 = torch.sum(torch.exp(-self.log_vars[0])*loss1 + self.log_vars[0], -1)
        loss2 = torch.sum(torch.exp(-self.log_vars[1])*loss2 + self.log_vars[1], -1)
        
        return [loss1, loss2], X, Y, Y[0, 0, 0]

    def fetch_minibatch(self):  # Generate time + a Brownian motion
        T = self.T

        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = np.zeros((M, N + 1, D))  # M x (N+1) x D

        dt = T / N

        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x D
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)

        return t, W

    def train(self, N_Iter, learning_rate):
        best_training_loss = 1e6; weights_path = 'allen_uncert.pth'
        loss_temp = np.array([])

        previous_it = 0
        if self.iteration != []:
            previous_it = self.iteration[-1]

        # Optimizers
        self.optimizer = PCGrad(optim.Adam(self.model.parameters(), lr=learning_rate))

        start_time = time.time()
        for it in range(previous_it, previous_it + N_Iter):
            self.optimizer.zero_grad()
            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D

            losses, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            loss = losses[0]
            if loss < best_training_loss:
                torch.save(self.model.state_dict(), weights_path)            

            self.optimizer.zero_grad()
            sum(losses).backward()
            self.optimizer.step()            

            loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss, Y0_pred, elapsed, learning_rate))
                start_time = time.time()

            # Loss
            if it % 100 == 0:
                self.training_loss.append(loss_temp.mean())
                loss_temp = np.array([])

                self.iteration.append(it)

            graph = np.stack((self.iteration, self.training_loss))
        return graph

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True
        losses, X_star, Y_star, Y0_pred = self.loss_function(t_star, W_star, Xi_star)
        return X_star, Y_star

    ###########################################################################
    ############################# Change Here! ################################
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        pass  # M x1

    @abstractmethod
    def g_tf(self, X):  # M x D
        pass  # M x 1
    
    @abstractmethod
    def aux_g_tf(self, X):  # M x D
        pass  # M x 1

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)  # M x D

    @abstractmethod
    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return torch.diag_embed(torch.ones([M, D])).to(self.device)  # M x D x D
    ###########################################################################