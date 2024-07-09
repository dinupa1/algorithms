import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


class OmniNet(nn.Module):
    def __init__(self, input_size=1, nodes=[30, 30], output_size=1):
        super(OmniNet, self).__init__()
        self.fc_stack = nn.Sequential()

        for i in range(len(nodes) + 1):
            self.fc_stack.add_module(f"fc_{i}", nn.Linear(input_size if i == 0 else nodes[i - 1], nodes[i] if i < len(nodes) else output_size))
            if i < len(nodes):
                self.fc_stack.add_module(f"relu_{i}", nn.ReLU())
        self.fc_stack.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.fc_stack(x)
        return x


class OmniLoss(nn.Module):
    def __init__(self):
        super(OmniLoss, self).__init__()

    def forward(self, outputs, targets, weights):
        loss_fn = nn.BCELoss(reduction="none")
        loss = weights * loss_fn(outputs, targets)
        return torch.mean(loss)


class OmniDataset(Dataset):
    def __init__(self, X_sim, X_data, W_sim, W_data):
        super(OmniDataset, self).__init__()

        self.X = np.concatenate([X_sim, X_data]).reshape(-1, 1)
        self.W = np.concatenate([W_sim, W_data]).reshape(-1, 1)
        self.Y = np.concatenate([np.zeros((len(X_sim), 1)), np.ones((len(X_data), 1))])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.W[idx], self.Y[idx]


class OmniFit:
    def __init__(self, train_dataloader, val_dataloader, model, optimizer, criterion, patience, device):
        super(OmniFit, self).__init__()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience

        print("=====> model summary ")
        print(f"using device {device}")
        print(self.model)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total trainable params: {total_trainable_params}")


        self.best_state = self.model.state_dict()
        self.best_epoch = None
        self.best_val_loss = None
        self.i_try = 0
        self.epoch = 0
        self.size = len(train_dataloader.dataset)


    def backpropagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_step(self):
        self.model.train()
        for batch, (inputs, weights, targets) in enumerate(self.train_dataloader):
            inputs, weights, targets = inputs.to(self.device).float(), weights.to(self.device).float(), targets.to(self.device).float()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets, weights)

            # Backpropagation
            self.backpropagation(loss)

            loss, current = loss.item(), (batch + 1) * len(inputs)
            print("\r" + f"[Epoch {self.epoch:>3d}] [{current:>5d}/{self.size:>5d}] [Train_loss: {loss:>7f}]", end="")

    def eval_step(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            for batch, (inputs, weights, targets) in enumerate(data_loader):
                inputs, weights, targets = inputs.to(self.device).float(), weights.to(self.device).float(), targets.to(self.device).float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, weights)
                auc = accuracy_score(targets.cpu().numpy().reshape(-1), outputs.cpu().numpy().round().reshape(-1), sample_weight=weights.cpu().numpy().reshape(-1))
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

            print("\r" + " " * (50), end="")
            print("\r" + f"[Epoch {epoch:>3d}] [Train_loss: {train_loss:>7f} Train_auc: {train_auc:>7f}] [Val_loss: {val_loss:>7f} Val_auc: {val_auc:>7f}]")

            if self.best_val_loss == None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.i_try = 0
            elif self.i_try < self.patience:
                self.i_try += 1
            else:
                print(f"Early stopping! Restore state at epoch {self.best_epoch}.")
                print(f"[Best_val_loss: {self.best_val_loss:>7f}]")
                self.model.load_state_dict(self.best_state)
                break

    def cal_weights(self, xvals):
        preds = self.model(torch.from_numpy(xvals).reshape(-1, 1).float().to(self.device)).cpu().detach().double().numpy().ravel()
        return preds/(1. - preds)