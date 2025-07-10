import torch

# ======================================
# ==========     for NN1      ==========
# ======================================

def binary_accuracy(output, target, threshold=0.5):
    '''
    Computes accuracy for binary classification based on a probability threshold.

    :param output: model output logits (no sigmoid applied), shape [B, 1]
    :param target: ground truth binary labels (0 or 1), shape [B]
    :param threshold: decision threshold, default = 0.5
    :return: accuracy as float (between 0 and 1)
    '''
    probs = torch.sigmoid(output) # to get probabilities from logit
    preds = (probs >= threshold).int().view(-1) # Convert probabilities to predicted class (0 or 1)
    target = target.view(-1).int()

    correct = (preds == target).sum().item()
    total = target.numel()

    return correct / total

# ======================================
# ==========     for AE      ===========
# ======================================

def reconstruction_error(output, target):
    return torch.mean(torch.abs(output - target)).item()

# ======================================
# ==========     for NN2      ==========
# ======================================

def multiclass_accuracy(output, target):
    preds = torch.argmax(output, dim=1)
    correct = (preds == target).sum().item()
    return correct / target.numel()

# ======================================

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


