import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from OmniFold import OmniNet, OmniLoss, OmniDataset, OmniFit

import uproot
import awkward as ak

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


nodes = [50, 50, 50, 50]
patience = 10
num_epochs = 2
learning_rate = 0.0001
batch_size = 1000

data_back = uproot.open("unfold.root:save")

X_data_reco = data_back["X_reco"].array().to_numpy()
W_data_true = data_back["W_true"].array().to_numpy()

sim_tree = uproot.open("unfold.root:sim")

X_sim_true = sim_tree["X_true"].array().to_numpy()
X_sim_reco = sim_tree["X_reco"].array().to_numpy()
W_sim_true = sim_tree["W_true"].array().to_numpy()
W_sim_true1 = sim_tree["W_true1"].array().to_numpy()
W_sim_reco1 = sim_tree["W_reco1"].array().to_numpy()


print("===> detector level ")

X_sim_train, X_sim_val, X_data_train, X_data_val, W_sim_train, W_sim_val, W_data_train, W_data_val = train_test_split(X_sim_reco, X_data_reco, W_sim_true1, W_data_true, test_size=0.3, shuffle=True)

train_dataset = OmniDataset(X_sim_train, X_data_train, W_sim_train, W_data_train)
val_dataset = OmniDataset(X_sim_val, X_data_val, W_sim_val, W_data_val)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = OmniNet(nodes=nodes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = OmniLoss()

model_fit = OmniFit(train_dataloader, val_dataloader, model, optimizer, criterion, patience, device)
model_fit.fit(num_epochs)

weights1 = model_fit.cal_weights(X_sim_reco)

W_sim_reco1 = W_sim_true1* weights1


print("===> particle level ")

X_sim_train, X_sim_val, X_data_train, X_data_val, W_sim_train, W_sim_val, W_data_train, W_data_val = train_test_split(X_sim_true, X_sim_true, W_sim_true, W_sim_reco1, test_size=0.3, shuffle=True)

train_dataset = OmniDataset(X_sim_train, X_data_train, W_sim_train, W_data_train)
val_dataset = OmniDataset(X_sim_val, X_data_val, W_sim_val, W_data_val)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = OmniNet(nodes=nodes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = OmniLoss()

model_fit = OmniFit(train_dataloader, val_dataloader, model, optimizer, criterion, patience, device)
model_fit.fit(num_epochs)

weights2 = model_fit.cal_weights(X_sim_true)

W_sim_true1 = weights2


outfile = uproot.recreate("unfold.root", compression=uproot.ZLIB(4))

outfile["sim"] = {
    "X_true": X_sim_true,
    "X_reco": X_sim_reco,
    "W_true": W_sim_true,
    "W_true1": W_sim_true1,
    "W_reco1": W_sim_reco1,
}

outfile["save"] = {
    "X_reco": X_data_reco,
    "W_true": W_data_true,
}

outfile.close()