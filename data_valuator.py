import numpy as np
import torch
import torch.nn as nn
from torch import optim
import math
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




class Data_Valuator_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Data_Valuator_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_units = hidden_size
        self.batch_size = batch_size
        self.num_layers = 1
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(self.hidden_size, 1, bias = True)

        self.apply(self._init_weights)

    def forward(self, x_input, hidden):
        
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size), hidden)
        # print(self.linear.weight)
        # print(self.linear.bias)
        return self.sigmoid(
            self.linear(self.hidden[1].view(self.batch_size,self.hidden_size))+
            self.linear(self.hidden[0].view(self.batch_size,self.hidden_size))
        ), self.linear(self.hidden[1].view(self.batch_size,self.hidden_size))+self.linear(self.hidden[0].view(self.batch_size,self.hidden_size))  
    
    def init_hidden(self):
        return (torch.ones(self.num_layers, self.batch_size, self.hidden_size),
                torch.ones(self.num_layers, self.batch_size, self.hidden_size))

    def _init_weights(self, module):
        for value in module.state_dict():
            param = module.state_dict()[value]
            if 'weight_ih' in value:
                print(value,param.shape,'Orthogonal')
                torch.nn.init.orthogonal_(module.state_dict()[value],gain=2)#input TO hidden ORTHOGONALLY || Wii, Wif, Wic, Wio
            elif 'weight_hh' in value:
                #INITIALIZE SEPERATELY EVERY MATRIX TO BE THE IDENTITY AND THE STACK THEM                        
                weight_hh_data_ii = torch.eye(self.hidden_units,self.hidden_units)#H_Wii
                weight_hh_data_if = torch.eye(self.hidden_units,self.hidden_units)#H_Wif
                weight_hh_data_ic = torch.eye(self.hidden_units,self.hidden_units)#H_Wic
                weight_hh_data_io = torch.eye(self.hidden_units,self.hidden_units)#H_Wio
                weight_hh_data = torch.stack([weight_hh_data_ii,weight_hh_data_if,weight_hh_data_ic,weight_hh_data_io], dim=0)
                weight_hh_data = weight_hh_data.view(self.hidden_units*4,self.hidden_units)
                print(value,param.shape,weight_hh_data.shape,self.hidden_units,'Identity')
                module.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY.state_dict()[value].data.copy_(weight_hh_data)#hidden TO 
            elif 'bias' in value:
                print(value,param.shape,'Zeros')
                torch.nn.init.constant_(module.state_dict()[value], val=0)
                module.state_dict()[value].data[self.hidden_units:self.hidden_units*2].fill_(1)#set the forget gate | (b_ii|b_if|b_ig|b_io)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.05, std=0.001)
            if module.bias is not None:
                module.bias.data.zero_()








def SelectionFromProb(prob, g):
    return torch.bernoulli(prob, generator = g)


def Data_Valuator_Train(model, log_pi, reward, alpha):
    log_pi.backward()
    for p in model.parameters():
        p.data.add(reward * p.grad.data, alpha=alpha)  #Beta = 0.001
        if p.grad is not None:
            p.grad.zero_()
    return 0

