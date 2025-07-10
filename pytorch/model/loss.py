import torch.nn as nn
import torch.nn.functional as F

# =====================================
# =========     for NN 1      =========
# =====================================

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_with_logits(output, target):
    '''
    Computes the binary cross-entropy loss with logits for binary classification.
    '''
    target = target.float().view_as(output)  # convert target to [B, 1]
    return nn.BCEWithLogitsLoss()(output, target.float())  # target must be float (0.0/1.0)

# ======================================
# ==========     for AE      ===========
# ======================================

def mse_loss(output, target):
    return nn.MSELoss()(output, target)

def mae_loss(output, target):
    return nn.L1Loss()(output, target)


# =====================================
# =========     for NN 2      =========
# =====================================

def cross_entropy_loss(output, target):
    return nn.CrossEntropyLoss()(output, target) # automatically applies Softmax, so no need for it in NN_2