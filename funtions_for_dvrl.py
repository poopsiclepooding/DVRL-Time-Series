import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


def SelectBatches(x, y, select):
    #Multiply inputs x and y with select vector to decide which inputs have been selected
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


def DataValuator_LogPi(selection_vector, prob_vector):
    log_pi_list = []
    for i in range(selection_vector.shape[0]):
        if torch.is_nonzero(selection_vector[i] == 1):
            log_pi_list.append(prob_vector[i].log())
        else:
            log_pi_list.append((1 - prob_vector[i]).log())
    log_pi = torch.sum(torch.stack(log_pi_list))
    return log_pi