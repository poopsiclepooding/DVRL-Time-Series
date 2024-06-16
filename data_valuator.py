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
                        nn.LeakyReLU(),
                        nn.Linear(hidden_size, 128, bias = True),
                        nn.LeakyReLU(),
                        nn.Linear(128, 64, bias = True),
                        nn.LeakyReLU(),
                        nn.Linear(64, 32, bias = True),
                        nn.LeakyReLU(),
                        nn.Linear(32, 16, bias = True),
                        nn.LeakyReLU(),
                        nn.Linear(16, 1, bias = True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x.to(torch.float32))
        return self.sigmoid(x), x



# class Data_Valuator_LSTM(nn.Module):
#     def __init__(self, batch_size, hidden_size=64, input_size=1):
#         super(Data_Valuator_LSTM, self).__init__()

#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.hidden_units = hidden_size
#         self.batch_size = batch_size
#         self.num_layers = 1
        
#         self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, proj_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.linear = nn.Linear(4, 1, bias = False)

#         # self.apply(self._init_weights)

#     def forward(self, x_input, hidden):
        
#         lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[1], x_input.shape[0], self.input_size), hidden)
#         out = self.sigmoid(
#             self.linear(lstm_out.squeeze(-1).transpose(1,0)[:,-5:-1])
#             # self.linear(self.hidden[1].squeeze(0))
#             # +
#             # self.linear(self.hidden[0].squeeze(0))
#         )
#         return out#, (1/10)*(self.linear(self.hidden[1].view(self.batch_size,self.hidden_size)))#+self.linear(self.hidden[0].view(self.batch_size,self.hidden_size)))  
    
#     def init_hidden(self, batch_size):
#         return (0.1*torch.ones(self.num_layers, batch_size, 1),
#                 0.1*torch.ones(self.num_layers, batch_size, self.hidden_size))

# #     def _init_weights(self, module):
# #         for value in module.state_dict():
# #             param = module.state_dict()[value]
# #             if 'weight_ih' in value:
# #                 print(value,param.shape,'Orthogonal')
# #                 torch.nn.init.normal_(module.state_dict()[value],mean=0.0, std=0.02)#input TO hidden ORTHOGONALLY || Wii, Wif, Wic, Wio
# #             elif 'weight_hh' in value:
# #                 #INITIALIZE SEPERATELY EVERY MATRIX TO BE THE IDENTITY AND THE STACK THEM                        
# #                 weight_hh_data_ii = 0.01*torch.ones(self.hidden_units,self.hidden_units)#H_Wii
# #                 weight_hh_data_if = 0.01*torch.ones(self.hidden_units,self.hidden_units)#H_Wif
# #                 weight_hh_data_ic = 0.01*torch.ones(self.hidden_units,self.hidden_units)#H_Wic
# #                 weight_hh_data_io = 0.01*torch.ones(self.hidden_units,self.hidden_units)#H_Wio
# #                 weight_hh_data = torch.stack([weight_hh_data_ii,weight_hh_data_if,weight_hh_data_ic,weight_hh_data_io], dim=0)
# #                 weight_hh_data = weight_hh_data.view(self.hidden_units*4,self.hidden_units)
# #                 print(value,param.shape,weight_hh_data.shape,self.hidden_units,'Identity')
# #                 module.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY.state_dict()[value].data.copy_(weight_hh_data)#hidden TO 
# #             elif 'bias' in value:
# #                 print(value,param.shape,'Zeros')
# #                 torch.nn.init.normal_(module.state_dict()[value], mean=0.0, std=0.05)
# #                 module.state_dict()[value].data[self.hidden_units:self.hidden_units*2].fill_(1)#set the forget gate | (b_ii|b_if|b_ig|b_io)
# #         if isinstance(module, nn.Linear):
# #             module.weight.data.normal_(mean=0.05, std=0.001)
# #             if module.bias is not None:
# #                 module.bias.data.fill_(0.5)




# def print_weights(module):
#     for value in module.state_dict():
#         param = module.state_dict()[value]
#         if 'weight_ih' in value:
#             print(value,param.shape,value.grad)
#         elif 'weight_hh' in value:
#             print(value,param.shape,value.grad)
#         elif 'bias' in value:
#             print(value,param.shape,value.grad)
#     if isinstance(module, nn.Linear):
#         print(module.weight.grad)
#         print(module.bias.grad)


# # def SelectionFromProb_2(prob_vector):
# #     select = np.random.binomial(1, prob_vector, prob_vector.shape)

# #     # Exception (When selection probability is 0)
# #     if np.sum(sel_prob_curr) == 0:
# #         prob_vector = 0.5 * np.ones(np.shape(prob_vector))
# #         select = np.random.binomial(1, prob_vector, prob_vector.shape)

# #     return select

# # def SelectionFromProb(prob, g):
# #     return torch.bernoulli(prob, generator = g)

# # def getBack(var_grad_fn):
# #     print(var_grad_fn)
# #     for n in var_grad_fn.next_functions:
# #         if n[0]:
# #             try:
# #                 tensor = getattr(n[0], 'variable')
# #                 print(n[0])
# #                 print('Tensor with grad found:', tensor)
# #                 print(' - gradient:', tensor.grad)
# #                 print()
# #             except AttributeError as e:
# #                 getBack(n[0])
# def Data_Valuator_Train_2(model, log_pi, reward, alpha):
#     log_pi.backward()
#     #getBack(log_pi.grad_fn)
#     for p in model.parameters():
#         p.data.add_(-p.grad.data, alpha=alpha)  #Beta = 0.001
#         if p.grad is not None:
#             p.grad.detach_()
#             p.grad.zero_()
#     return 0



# def Data_Valuator_Train(model, log_pi, reward, alpha):
#     log_pi.backward()
#     #getBack(log_pi.grad_fn)
#     for p in model.parameters():
#         p.data.add_((reward.detach()) * p.grad.data, alpha=alpha)  #Beta = 0.001
#         if p.grad is not None:
#             p.grad.detach_()
#             p.grad.zero_()
#     return 0


