import torch


def get_thres(neg_pairs, far=1e-1):
    n = neg_pairs.numel()
    nfa = int(n * far)
    return neg_pairs.topk(nfa + 1, largest=True).values[-1].item()


def tar(pos_pairs, thres):
    return [(pos_pairs>th).float().mean().item() for th in thres]


def eval_far_tar(pos_pairs, neg_pairs, fars=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
    thres = [get_thres(neg_pairs, far) for far in fars]
    tars = tar(pos_pairs, thres)
    return thres, tars



if __name__ == '__main__':
    pos_pairs = torch.randn(100**2)+1
    neg_pairs = torch.randn(100**2)-1

    print(eval_far_tar(pos_pairs, neg_pairs))

