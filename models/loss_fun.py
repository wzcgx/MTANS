import torch.nn.functional as F
import torch

from torch.autograd import Function
from itertools import repeat
import numpy as np



# for metric code: https://github.com/MIC-DKFZ/BraTS2017/blob/master/utils_validation.py

cross_entropy = F.cross_entropy

def hard_cross_entropy(output, target, alpha=3.0):
    mtx = F.cross_entropy(output, target, reduce=False)

    bg = (target == 0)

    neg = mtx[bg]
    pos = mtx[1-bg]

    Np, Nn = pos.numel(), neg.numel()

    pos = pos.sum()

    k = min(Np*alpha, Nn)
    if k > 0:
        neg, _ = torch.topk(neg, int(k))
        neg = neg.sum()
    else:
        neg = 0.0

    loss = (pos + neg)/ (Np + k)

    return loss


def hard_per_im_cross_entropy(output, target, alpha=3.0):
    n, c = output.shape[:2]
    output = output.view(n, c, -1)
    target = target.view(n, -1)

    mtx = F.cross_entropy(output, target, reduce=False)

    pos = target > 0
    num_pos = pos.long().sum(dim=1, keepdim=True)

    loss = mtx.clone().detach()
    loss[pos] = 0
    _, loss_idx = loss.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)

    num_neg = torch.clamp(alpha*num_pos, max=pos.size(1)-1)
    neg = idx_rank < num_neg

    return mtx[neg + pos].mean()


def focal_loss(output, target, alpha=0.25, gamma=2.0):
    n = target.size(0)

    lsfm = F.cross_entropy(output, target, reduce=False)

    pos = (target > 0).float()
    Np  = pos.view(n, -1).sum(1).view(n, 1, 1, 1)

    Np  = torch.clamp(Np, 1.0)
    z   = pos * alpha / Np / n  + (1.0 - pos) * (1.0 - alpha) / Np / n
    z   = z.detach()

    focal = torch.pow(1.0 - torch.exp(-lsfm), gamma) * lsfm * z

    return focal.sum()


def mean_cross_entropy(output, target, alpha=3.0):
    mtx = F.cross_entropy(output, target, reduce=False)

    bg = (target == 0)

    neg = mtx[bg]
    pos = mtx[1-bg]

    pos = pos.mean() if pos.numel() > 0 else 0
    neg = neg.mean() if pos.neg() > 0 else 0

    loss = (neg * alpha + pos)/(alpha + 1.0)
    return loss




eps = 0.1
def dice(output, target):
    num = 2*(output*target).sum() + eps
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den


def cross_entropy_dice(output, target, weight=1.0):
    loss = weight * F.cross_entropy(output, target)
    output = F.softmax(output, dim=1)
    for c in range(1, 4):
        o = output[:, c]
        t = (target==c).float()
        loss += 0.25*dice(o, t)

    return loss


def cross_entropy_dice_binary(output, target, weight=1.0):
    loss = weight * F.cross_entropy(output, target)
    output = F.softmax(output, dim=1)
    for c in range(1, 2):
        o = output[:, c]
        t = (target==c).float()
        loss += 0.25*dice(o, t)

    return loss

# in original paper: class 3 is ignored
# https://github.com/MIC-DKFZ/BraTS2017/blob/master/dataset.py#L283
# dice score per image per positive class, then aveg
def dice_per_im(output, target):
    n = output.shape[0]
    output = output.view(n, -1)
    target = target.view(n, -1)
    num = 2*(output*target).sum(1) + eps
    den = output.sum(1) + target.sum(1) + eps
    return 1.0 - (num/den).mean()

def cross_entropy_dice_per_im(output, target, weight=1.0):
    loss = weight * F.cross_entropy(output, target)
    output = F.softmax(output, dim=1)
    for c in range(1, 4):
        o = output[:, c]
        t = (target==c).float()
        loss += 0.25*dice_per_im(o, t)

    return loss

def seg_AN_dice_loss(input,target):
    num=input*target
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)

    den1=input*input
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)

    den2=target*target
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)

    dice=2*(num/(den1+den2))

    dice_total=1-1*torch.sum(dice)/dice.size(0)#divide by batchsize

    return dice_total


class DiceLoss(Function):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, input, target, save=True):
        if save:
            self.save_for_backward(input, target)
        eps = 0.000001
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
#       print(input)
        intersect = torch.dot(result, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
            union, intersect, target_sum, result_sum, 2*IoU))
        out = torch.FloatTensor(1).fill_(2*IoU)
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_input , None

def dice_loss(input, target):
    return DiceLoss()(input, target)

def dice_error(input, target):
    eps = 0.000001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    intersect = torch.dot(result, target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + 2*eps
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#        union, intersect, target_sum, result_sum, 2*IoU))
    return 2*IoU