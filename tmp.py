# %%
import poutyne as pt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# %%
x = torch.rand(120, 4)
y = torch.rand(120)

# %%
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(y)

    def __getitem__(self, idx):
        
        return x[idx], y[idx]

# %%
dataset = CustomDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32)

# %%
network = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(), 
    nn.Linear(10, 1)
)

# %%
model = pt.Model(network, "Adam", 'mse')

# %%
model.fit_generator(dataloader, epochs=10)
# %%
