import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_f=4, h1=8, h2=8, out_f=3):
        super().__init__()
        self.fc1 = nn.Linear(in_f, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_f)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


torch.manual_seed(41)

model = Model()

import pandas as pd
import matplotlib.pyplot as plt

#%matplotlib inline

url = 'https://gist.githubusercontent.com/netj/8836201/raw/iris.csv'
my_df = pd.read_csv(url)
print(my_df.tail())

print(my_df['variety'])