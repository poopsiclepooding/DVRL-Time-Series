import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F



class Data_Valuator_MLP(nn.Module):
    def __init__(self, sequence_length, hidden_size):
        "d_model is same as lookback"
        super(Data_Valuator_MLP, self).__init__()

        self.layer = nn.Sequential(
                        nn.Linear(sequence_length, hidden_size, bias = True),
                        nn.ReLU(),
                        nn.Linear(hidden_size, 128, bias = True),
                        nn.ReLU(),
                        nn.Linear(128, 64, bias = True),
                        nn.ReLU(),
                        nn.Linear(64, 32, bias = True),
                        nn.ReLU(),
                        nn.Linear(32, 16, bias = True),
                        nn.ReLU(),
                        nn.Linear(16, 1, bias = True),
                        nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x.to(torch.float32))



def SelectionFromProb(prob, g):
    return torch.bernoulli(prob, generator = g)


def Data_Valuator_Train(model, log_pi, reward):
    log_pi.backward()
    for p in model.parameters():
        p.data.add(reward * p.grad.data, alpha=0.5)  #Beta = 0.001
        if p.grad is not None:
            p.grad.zero_()
    return 0

