import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def softmax(output, target):
    loss = nn.CrossEntropyLoss()
    return loss(output, target)