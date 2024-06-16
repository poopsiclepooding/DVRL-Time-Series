import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


def SelectBatches(x, y, select):   
    '''        
    : param x:                          x input of size (batch_size, lookback)
    : param y:                          y input of size (batch_size, lookahead)
    : param select:                     vector to select x and y of size (batch_size, 1)
    : return x_batch, y_batch           x_batch is the selected x batches
    :                                   y_batch is the selected y batches

    '''
    x_bat_zero = x*select
    y_bat_zero = y*select

    #Remove the zero input which occur due to above multiplication
    x_non_empty_mask = x_bat_zero.abs().sum(dim=1).bool()
    y_non_empty_mask = y_bat_zero.abs().sum(dim=1).bool()

    #Find what this does
    x_batch = x_bat_zero[x_non_empty_mask,:]
    y_batch = y_bat_zero[y_non_empty_mask,:]

    return x_batch, y_batch



def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    '''        
    : param dataset:                    x input of size (batch_size, lookback)
    : param y:                          y input of size (batch_size, lookahead)
    : param select:                     vector to select x and y of size (batch_size, 1)
    : return x_batch, y_batch           x_batch is the selected x batches
    :                                   y_batch is the selected y batches

    '''
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)



def MovingBatch(x_train, y_train, time_window_size, T):
    return x_train[T:T+time_window_size], y_train[T:T+time_window_size]


def MAL(val_loss, T, prev_avg):
    return ((T-1.0)/T)*prev_avg + (1/T)*val_loss

def HMM(val_loss, T, asymtote):
    return val_loss - asymtote

def IIR(val_loss, T, prev_avg, beta):
    return beta*val_loss + (1-beta)*prev_avg




def SelectionFromProb_2(prob_vector):
    prob_vector = prob_vector.to("cpu")
    prob_vector = prob_vector.detach().numpy()
    select = np.random.binomial(1, prob_vector, prob_vector.shape)

    # Exception (When selection probability is 0)
    if np.sum(select) == 0:
        prob_vector = 0.5 * np.ones(np.shape(prob_vector))
        select = np.random.binomial(1, prob_vector, prob_vector.shape)

    return torch.from_numpy(select)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


def DataValuator_LogPi_2(selection_vector, prob_vector, epsilon, threshold, reward, explore_parameter):
    reward = reward.detach()
    selection_vector = selection_vector.detach()
    if reward < 0:
        log_prob =  2*prob_vector * selection_vector #+ (1 - prob_vector) * (1 - selection_vector)
    
    if reward >= 0:
        log_prob =   (1 - prob_vector) * selection_vector #+  prob_vector * (1 - selection_vector) 
    prob = torch.sum(log_prob, dim=0)

    dve_loss = abs(reward)*prob + explore_parameter * (torch.maximum(torch.mean(prob_vector) - threshold, torch.tensor(0)) + torch.maximum((1-threshold) - torch.mean(prob_vector), torch.tensor(0)))
    return dve_loss
