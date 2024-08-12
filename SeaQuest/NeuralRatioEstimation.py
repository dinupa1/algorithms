import numpy as np
import matplotlib.pyplot as plt

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rc("font", size=14)


class NRENet(nn.Module):
    def __init__(self, input_size=2, nodes=[30, 30], dropouts=[0, 0.1, 0], output_size=1):
        super(NRENet, self).__init__()
        self.fc_stack = nn.Sequential()

        for i in range(len(nodes) + 1):
            self.fc_stack.add_module(f"fc_{i}", nn.Linear(input_size if i == 0 else nodes[i - 1], nodes[i] if i < len(nodes) else output_size))
            if i < len(nodes):
                self.fc_stack.add_module(f"relu_{i}", nn.ReLU())
                # self.fc_stack.add_module(f"batchNorm_{i}", nn.BatchNorm1d(nodes[i]))
            if dropouts[i] > 0:
                self.fc_stack.add_module(f"dropout_{i}", nn.Dropout(p=dropouts[i]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, theta):
        inputs = torch.cat((x, theta), dim=-1).float()
        log_ratio = self.fc_stack(inputs)
        outputs = self.sigmoid(log_ratio)
        return outputs, log_ratio


class NREDataset(Dataset):
    def __init__(self, X_data, T_data):
        super(NREDataset, self).__init__()

        self.X = X_data.reshape(-1, 1)
        self.T = T_data.reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]


class NREFit:
    def __init__(self, train_dataloader, val_dataloader, model, optimizer, criterion, patience, device):
        super(NREFit, self).__init__()

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
        self.best_auc = None
        self.i_try = 0
        self.epoch = 0
        self.size = len(train_dataloader.dataset)

    def backpropagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_step(self):
        num_iterations = len(self.train_dataloader)//2
        loader = iter(self.train_dataloader)
        self.model.train()
        for batch_idx in range(num_iterations):
            x_a, theta_a = next(loader)
            x_b, theta_b = next(loader)

            x_a, theta_a = x_a.to(self.device), theta_a.to(self.device)
            x_b, theta_b = x_b.to(self.device), theta_b.to(self.device)

            y_dep_a, _ = self.model(theta_a, x_a)
            y_ind_a, _ = self.model(theta_b, x_a)
            y_dep_b, _ = self.model(theta_b, x_b)
            y_ind_b, _ = self.model(theta_b, x_a)

            loss_a = self.criterion(y_dep_a, torch.ones_like(y_dep_a)) + self.criterion(y_ind_a, torch.zeros_like(y_ind_a))
            loss_b = self.criterion(y_dep_b, torch.ones_like(y_dep_b)) + self.criterion(y_ind_b, torch.zeros_like(y_ind_b))

            loss = loss_a + loss_b

            # Backpropagation
            self.backpropagation(loss)

            loss, current = loss.item(), (batch_idx + 1) * len(x_a)
            print("\r" + f"[Epoch {self.epoch:>3d}] [{current:>5d}/{self.size:>5d}] [Train_loss: {loss:>7f}]", end="")

    def eval_step(self, data_loader):
        num_iterations = len(data_loader)//2
        loader = iter(data_loader)
        self.model.eval()
        with torch.no_grad():
            for batch_idx in range(num_iterations):
                x_a, theta_a = next(loader)
                x_b, theta_b = next(loader)

                x_a, theta_a = x_a.to(self.device), theta_a.to(self.device)
                x_b, theta_b = x_b.to(self.device), theta_b.to(self.device)

                y_dep_a, _ = self.model(theta_a, x_a)
                y_ind_a, _ = self.model(theta_b, x_a)
                y_dep_b, _ = self.model(theta_b, x_b)
                y_ind_b, _ = self.model(theta_a, x_b)

                loss_a = self.criterion(y_dep_a, torch.ones_like(y_dep_a)) + self.criterion(y_ind_a, torch.zeros_like(y_ind_a))
                loss_b = self.criterion(y_dep_b, torch.ones_like(y_dep_b)) + self.criterion(y_ind_a, torch.zeros_like(y_ind_a))

                loss = loss_a + loss_b

                outputs = torch.cat((y_dep_a, y_ind_a, y_dep_b, y_ind_b))
                targets = torch.cat((torch.ones_like(y_dep_a), torch.zeros_like(y_ind_a), torch.ones_like(y_dep_b), torch.zeros_like(y_ind_b)))

                auc = roc_auc_score(targets.cpu().numpy().reshape(-1), outputs.cpu().numpy().reshape(-1))
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
                self.best_auc = val_auc
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.i_try = 0
            elif self.i_try < self.patience:
                self.i_try += 1
            else:
                print(f"Early stopping! Restore state at epoch {self.best_epoch}.")
                print(f"[Best_val_loss: {self.best_val_loss:>7f}, auc: {self.best_auc:>7f}]")
                self.model.load_state_dict(self.best_state)
                break

    def sampler(self, xvals, samples=10, mu_init=0., prop_std=0.5, burn=0.2):
        size =xvals.shape[0]
        mu_current = mu_init
        posterior = []
        for i in range(samples):
            mu_proposal = np.random.normal(mu_current, prop_std)
            X_tensor = torch.from_numpy(xvals).reshape(-1, 1).to(self.device)
            mu_current_tensor = mu_current* torch.ones_like(X_tensor).to(self.device)
            mu_proposal_tensor = mu_proposal* torch.ones_like(X_tensor).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs, log_ratio_current = self.model(mu_current_tensor, X_tensor)
                outputs, log_ratio_proposal = self.model(mu_proposal_tensor, X_tensor)
                sum_prob_density_proposal = np.sum(log_ratio_proposal.cpu().detach().double().numpy().ravel())
                sum_prob_density_current = np.sum(log_ratio_current.cpu().detach().double().numpy().ravel())
                prob_accept = min(0., sum_prob_density_proposal - sum_prob_density_current)
                accept = np.log(np.random.rand()) < prob_accept
                if accept:
                    mu_current = mu_proposal

                if i > int(burn* samples):
                    posterior.append(mu_current)
        return np.array(posterior)

N_data = 100000
mu_sim = 0.
sigma = 1.
mu_data = 0.2

p_x = np.random.normal(mu_sim, sigma, N_data)
p_mu = np.random.uniform(-2., 2., N_data)
p_x_mu = np.array([np.random.normal(mu, sigma) for mu in p_mu])

# bins = np.linspace(-5., 5., 31)
# plt.figure(figsize=(8., 8.))
# plt.hist(p_x, bins=bins, histtype="step", density=True)
# plt.xlabel("x")
# plt.ylabel(r"$p(x)$")
# plt.savefig("plots/p_x.png")
# plt.close("all")

# bins = np.linspace(-2., 2., 31)
# plt.figure(figsize=(8., 8.))
# plt.hist(p_mu, bins=bins, histtype="step", density=True)
# plt.xlabel(r"$\mu$")
# plt.ylabel(r"$p(\mu)$")
# plt.savefig("plots/p_mu.png")
# plt.close("all")

# bins = np.linspace(-6., 6., 31)
# plt.figure(figsize=(8., 8.))
# plt.hist(p_x_mu, bins=bins, histtype="step", density=True)
# plt.xlabel("x")
# plt.ylabel(r"$p(x|\mu)$")
# plt.savefig("plots/p_x_mu.png")
# plt.close("all")


dataset = NREDataset(p_x_mu, p_mu)

train_dataset, val_dataset = random_split(dataset, [0.7, 0.3])

train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

nodes = [50, 50, 50, 50]
dropouts = [0., 0.2, 0., 0., 0., 0.]
patience = 10
epochs = 1000

model = NRENet(nodes=nodes, dropouts=dropouts)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

fit_model = NREFit(train_loader, val_loader, model, optimizer, criterion, patience, device)
fit_model.fit(epochs)

X_data = np.random.normal(mu_data, sigma, 500)

# bins = np.linspace(-6, 7., 31)
# plt.figure(figsize=(8., 8.))
# plt.hist(X_data, bins=bins, histtype="step", density=True)
# plt.xlabel("x")
# plt.ylabel("$counts$")
# plt.savefig("plots/data.png")
# plt.close("all")

posterior = fit_model.sampler(X_data, samples=10000)

mu_mean = np.mean(posterior)
mu_std = np.std(posterior)

print(f"---> length: {len(posterior)} mean: {mu_mean} std: {mu_std}")

# bins = np.linspace(-2., 2., 31)
plt.figure(figsize=(8., 8.))
plt.hist(posterior, bins=30, histtype="step", density=True)
plt.axvline(x=mu_data, linestyle="--", color="r", label=r"$\mu_{true}$")
plt.xlabel(r"$\mu$")
plt.ylabel(r"$p(\mu|x)$")
plt.text(0.7, 0.1, f"$\mu_{{fit}}$ = {mu_mean:.3f} +/- {mu_std:.3f}")
plt.legend(frameon=False)
plt.savefig("plots/p_mu_x.png")
plt.close("all")
# plt.show()
