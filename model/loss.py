import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def softmax(output, target):
    loss = nn.CrossEntropyLoss()
    return loss(output, target)

def knowledge_distillation_loss(logits, target, teacher_logits):
        alpha = 0.3
        T = 1
        
        student_loss = F.cross_entropy(input=logits, target=target)
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (T * T)
        total_loss =  (1. - alpha)*student_loss + alpha*distillation_loss

        return total_loss