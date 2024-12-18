import torch
import torch.nn as nn

class FLoss(nn.Module):
    """ Code acompanying the paper https://arxiv.org/abs/1805.07567

    :param beta: the beta parameter of fmeasure
    :type client: float
    """

    def __init__(self, beta=0.3):
        super(FLoss, self).__init__()
        self.beta = beta

    def forward(self, prediction, target, weight=None):
        assert prediction.min() >= 0 and prediction.max() <= 1, "min %f v.s. max %f" % (prediction.min(), prediction.max())

        EPS = 1e-6

        if weight is not None:
            prediction = prediction * weight
            target = target * weight

        N = prediction.size(0)

        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)

        floss  = (1 - fmeasure).mean()

        return floss
