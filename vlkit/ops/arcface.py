import math, torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAngularMargin(nn.Linear):
    """
    ArcFace https://arxiv.org/pdf/1801.07698
    """
    def __init__(self, in_features, out_features, s=32, m=0.5, warmup_iters=-1):

        super(AdditiveAngularMargin, self).__init__(in_features, out_features, bias=False)
        self.s = s
        self.m = m
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.warmup_iters = warmup_iters
        self.iter = 0

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=0.01)

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1, 1)

        if label is None or self.m == 0:
            return cosine * self.s

        if self.warmup_iters > 0:
            self.iter = self.iter + 1
            if self.iter < self.warmup_iters:
                m = (1 - math.cos((math.pi / self.warmup_iters) * self.iter)) / 2 * self.m
                if self.iter % (self.warmup_iters // 10) == 0:
                    print("ArcFace: iter %d, m=%.3e" % (self.iter, m))
            else:
                m = self.m
            self.cos_m = math.cos(m)
            self.sin_m = math.sin(m)
        else:
            m = self.m

        # sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # psi = cos(theta + m)
        psi_theta = cosine * self.cos_m - sine * self.sin_m
        # see http://data.kaizhao.net/notebooks/arcface-psi.html
        psi_theta = torch.where(cosine > -self.cos_m, psi_theta, -psi_theta - 2)

        onehot = torch.zeros_like(cosine).bool()
        onehot = onehot.scatter(dim=1, index=label.view(-1, 1).long(), value=1)
        output = torch.where(onehot, psi_theta, cosine) * self.s

        return output, cosine * self.s

    def __str__(self):
        return "ArcFace(in_features=%d out_features=%d s=%.3f m=%.3f warmup_iters=%d)" % \
               (self.weight.shape[1], self.weight.shape[0], self.s, self.m, self.warmup_iters)
