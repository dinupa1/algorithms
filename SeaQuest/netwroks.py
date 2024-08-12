import numpy as np
import matplotlib.pyplot as plt

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import uproot
from sklearn.covariance import log_likelihood
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split


def roc_auc(input, target, weight=None):
    fpr, tpr, _ = roc_curve(target, input, sample_weight=weight)
    tpr, fpr = np.array(list(zip(*sorted(zip(tpr, fpr)))))
    return 1 - auc(tpr, fpr)


def weight_fn(theta_0, theta_1, theta_2, phi, costh):
    weight = 1. + theta_0 * costh * costh + 2. * theta_1 * costh * torch.sqrt(1. - costh * costh) * torch.cos(phi) + 0.5 * theta_2 * (1. - costh * costh) * torch.cos(2. * phi)
    return weight/(1. + costh * costh)


class Network_R0(nn.Module):
    def __init__(self, mean_parsms=None, std_params=None, input_dim=8, output_dim=1, nodes=[30, 30], dropouts=[0.2, 0.2], activation=nn.ReLU()):
        super(Network_R0, self).__init__()

        self.fc_input = nn.Linear(input_dim, nodes[0], bias=True)
        self.fc_activation = activation

        self.fc_stack = nn.Sequential()

        for i in range(len(nodes) - 1):
            self.fc_stack.add_module(f"fc_linear_{i}", nn.Linear(nodes[i], nodes[i + 1], bias=True))
            self.fc_stack.add_module(f"fc_batchnorm_{i}", nn.BatchNorm1d(nodes[i + 1]))
            self.fc_stack.add_module(f"fc_activation_{i}", activation)
            if dropouts[i] > 0:
                self.fc_stack.add_module(f"fc_dropout_{i}", nn.Dropout(p=dropouts[i]))

        self.fc_output = nn.Linear(nodes[-1], output_dim, bias=True)
        self.fc_sigmoid = nn.Sigmoid()

    def forward(self, x, theta):
        x = torch.cat((x, theta), dim=-1)
        x = self.log_likelihood_ratio(x)
        x = self.fc_sigmoid(x)
        return x/(1 - x + 0.00000000000000000001)


class Network_T0(nn.Module):
    def __init__(self, init_thetas=[1., 0., 0.]):
        super(Network_T0, self).__init__()

        self.thetas = nn.Parameter(torch.tensor(init_thetas), requires_grad=True)

    def forward(self):
        return self.thetas


class Dataset_R0(Dataset):
    def __init__(self, X0, T0, R0, X1, T1, R1):
        super(Dataset_R0, self).__init__()

        self.X = torch.cat((X0, X1))
        self.T = torch.cat((T0, T1))
        self.R = torch.cat((R0, R1)).reshape(-1, 1)
        self.Y = torch.cat((torch.zeros(len(X0)), torch.ones(len(X1)))).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.R[idx], self.Y[idx]


class Dataset_T0(Dataset):
    def __init__(self, X0, R0, X1, R1):
        super(Dataset_T0, self).__init__()

        self.X = torch.cat((X0, X1))
        self.R = torch.cat((R0, R1)).reshape(-1, 1)
        self.Y = torch.cat((torch.zeros(len(X0)), torch.ones(len(X1)))).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.R[idx], self.Y[idx]


class Fit_R0:
    def __init__(self, train_dataloader, val_dataloader, network, optimizer, scheduler, criterion, patience, device):
        super(Fit_R0, self).__init__()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.network = network.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

        print("=====> Network summary ")
        print(f"using device {device}")
        print(self.network)
        total_trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f"total trainable params: {total_trainable_params}")

        self.best_state = self.network.state_dict()
        self.best_epoch = None
        self.best_val_loss = None
        self.best_auc = None
        self.i_try = 0
        self.epoch = 0
        self.size = len(train_dataloader.dataset)

    def backpropagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_step(self):
        self.network.train()
        for batch, (inputs, theta, weights, targets) in enumerate(self.train_dataloader):
            inputs, theta, weights, targets = inputs.float().to(self.device), theta.float().to(self.device), weights.float().to(self.device), targets.float().to(self.device)

            ratio = self.network(inputs, theta)
            logit = ratio/(ratio + 1)
            loss = self.criterion(logit, targets, weights)

            # Backpropagation
            self.backpropagation(loss)

            loss, current = loss.item(), (batch + 1) * len(inputs)
            print("\r" + f"[Epoch {self.epoch:>3d}] [{current:>5d}/{self.size:>5d}] [Train_loss: {loss:>5f}]", end="")

    def eval_step(self, data_loader):
        self.network.eval()
        with torch.no_grad():
            for batch, (inputs, theta, weights, targets) in enumerate(data_loader):
                inputs, theta, weights, targets = inputs.float().to(self.device), theta.float().to(self.device), weights.float().to(self.device), targets.float().to(self.device)

                ratio = self.network(inputs, theta)
                logit = ratio / (ratio + 1)
                loss = self.criterion(logit, targets, weights)

                auc = roc_auc(targets.cpu().numpy().reshape(-1), logit.cpu().numpy().reshape(-1), weight=weights.cpu().numpy().reshape(-1))

        return loss, auc

    def fit(self, num_epochs):

        for epoch in range(1, num_epochs+1):
            self.epoch = epoch

            # train
            self.train_step()

            # evaluate loss for traing set
            train_loss, train_auc = self.eval_step(self.train_dataloader)

            # evaluate loss for validation set
            val_loss, val_auc = self.eval_step(self.val_dataloader)

            self.train_losses.append(train_loss.item())
            self.val_losses.append(val_loss.item())
            self.epochs.append(self.epoch)

            print("\r" + " " * (50), end="")
            print("\r" + f"[Epoch {epoch:>3d}] [Train_loss: {train_loss:>7f} Train_auc: {train_auc:>7f}] [Val_loss: {val_loss:>7f} Val_auc: {val_auc:>7f}]")

            if self.best_val_loss == None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_auc = val_auc
                self.best_state = copy.deepcopy(self.network.state_dict())
                self.best_epoch = epoch
                self.i_try = 0
            elif self.i_try < self.patience:
                self.i_try += 1
            else:
                print(f"Early stopping! Restore state at epoch {self.best_epoch}.")
                print(f"[Best_val_loss: {self.best_val_loss:>7f}, Best_auc: {self.best_auc:>7f}]")
                self.network.load_state_dict(self.best_state)
                break

            self.scheduler.step()

    def history(self):
        return np.array(self.train_losses), np.array(self.val_losses), np.array(self.epochs)


class Fit_T0:
    def __init__(self, train_dataloader, val_dataloader, network, theta_network, optimizer, scheduler, criterion, patience, device):
        super(Fit_T0, self).__init__()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.network = network.to(device)
        self.theta_network = theta_network.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.theta_0 = []
        self.theta_1 = []
        self.theta_2 = []

        self.network.eval()

        print("=====> Network summary ")
        print(f"using device {device}")
        print(self.network)
        total_trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f"total trainable params: {total_trainable_params}")
        total_trainable_params = sum(p.numel() for p in self.theta_network.parameters() if p.requires_grad)
        print(f"total trainable params: {total_trainable_params}")
        print(f"Initial theta values : {self.theta_network()[0].item(), self.theta_network()[1].item(), self.theta_network()[2].item()}")

        self.best_state = self.theta_network.state_dict()
        self.best_epoch = None
        self.best_val_loss = None
        self.best_auc = None
        self.i_try = 0
        self.epoch = 0
        self.size = len(train_dataloader.dataset)

    def backpropagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_step(self):
        self.theta_network.train()
        for batch, (inputs, weights, targets) in enumerate(self.train_dataloader):
            inputs, weights, targets = inputs.float().to(self.device), weights.float().to(self.device), targets.float().to(self.device)

            thetas = self.theta_network()
            ones = torch.ones(len(inputs), len(thetas)).float().to(self.device)

            ratio = self.network(inputs, thetas * ones)
            logit = ratio / (ratio + 1)
            loss = self.criterion(logit, targets, weights)

            # Backpropagation
            self.backpropagation(loss)

            loss, current = loss.item(), (batch + 1) * len(inputs)
            print("\r" + f"[Epoch {self.epoch:>3d}] [{current:>5d}/{self.size:>5d}] [Train_loss: {loss:>5f}]", end="")

    def eval_step(self, data_loader):
        self.theta_network.eval()
        with torch.no_grad():
            for batch, (inputs, theta, weights, targets) in enumerate(data_loader):
                inputs, weights, targets = inputs.float().to(self.device), weights.float().to(self.device), targets.float().to(self.device)

                thetas = self.theta_network()
                ones = torch.ones(len(inputs), len(thetas)).float().to(self.device)

                ratio = self.network(inputs, thetas * ones)
                logit = ratio / (ratio + 1)
                loss = self.criterion(logit, targets, weights)

                auc = roc_auc(targets.cpu().numpy().reshape(-1), logit.cpu().numpy().reshape(-1), weight=weights.cpu().numpy().reshape(-1))

        return loss, auc, thetas

    def fit(self, num_epochs):

        for epoch in range(1, num_epochs+1):
            self.epoch = epoch

            # train
            self.train_step()

            # evaluate loss for traing set
            train_loss, train_auc, train_thetas = self.eval_step(self.train_dataloader)

            # evaluate loss for validation set
            val_loss, val_auc, val_thetas = self.eval_step(self.val_dataloader)

            self.train_losses.append(train_loss.item())
            self.val_losses.append(val_loss.item())
            self.epochs.append(self.epoch)

            self.theta_0.append(val_thetas[0].item())
            self.theta_1.append(val_thetas[1].item())
            self.theta_2.append(val_thetas[2].item())

            print("\r" + " " * (50), end="")
            print("\r" + f"[Epoch {epoch:>3d}] [Train_loss: {train_loss:>7f} Train_auc: {train_auc:>7f}] [Val_loss: {val_loss:>7f} Val_auc: {val_auc:>7f}]")

            if self.best_val_loss == None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_auc = val_auc
                self.best_state = copy.deepcopy(self.theta_network.state_dict())
                self.best_epoch = epoch
                self.i_try = 0
            elif self.i_try < self.patience:
                self.i_try += 1
            else:
                print(f"Early stopping! Restore state at epoch {self.best_epoch}.")
                self.theta_network.load_state_dict(self.best_state)
                thetas = self.theta_network()
                print(f"[Best_val_loss: {self.best_val_loss:>7f}, Best_val_auc: {self.best_auc:>7f}, Best_thetas: {thetas[0].item():>3f}, {thetas[1].item():>3f}, {thetas[2].item():>3f}]")
                break

            self.scheduler.step()

    def history(self):
        return np.array(self.train_losses), np.array(self.val_losses), np.array(self.epochs), np.array(self.theta_0), np.array(self.theta_1), np.array(self.theta_2)

    def best_thetas(self):
        thetas = self.theta_network()
        return thetas[0].item(), thetas[1].item(), thetas[2].item()


class Loss_R0(nn.Module):
    def __init__(self):
        super(Loss_R0, self).__init__()

    def forward(self, outputs, targets, weights):
        weighted_loss = targets* weights* torch.log(outputs) + (1 - targets)* weights* torch.log(1 - outputs)
        return -1. * torch.mean(weighted_loss)