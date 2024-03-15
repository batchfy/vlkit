import torch
import torch.nn.functional as F

def kl_loss(x, y, t=1):
    """
    kl loss which is often used in distillation
    """
    return kl_divergence(x/t, y/t) * (t**2)

def js_loss(x, y, t):
    """
    js loss, similar to kl_loss
    """
    return js_divergence(x/t, y/t) * (t**2)

def kl_divergence(x, y, normalize=True):
    """
    KL divergence between vectors
    When normalize = True, inputs x and y are vectors BEFORE normalization (eg. softmax),
    when normalize = False, x, y are probabilities that must sum to 1 
    """
    if normalize:
        x =  F.log_softmax(x, dim=1)
        y = F.softmax(y, dim=1)
    else:
        x = x.log()

    return F.kl_div(x, y, reduction="batchmean")

def js_divergence(x, y):
    """
    The Jensen–Shannon divergence
    Inputs are similar to kl_divergence
    """
    return 0.5 * kl_divergence(x, y) + 0.5 * kl_divergence(y, x)

