import torch
import warnings
warnings.filterwarnings('ignore')
import sklearn

LABEL_LIST = [
    "hate",
    "offensive",
    "none"
]

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def macro_f1(output, target):
    label_indices = list(range(len(LABEL_LIST)))
    return sklearn.metrics.f1_score(target, output, average="macro", labels=label_indices) * 100.0
