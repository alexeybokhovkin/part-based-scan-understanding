from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import numpy as np


def linear_assignment(distance_mat, row_counts=None, col_counts=None):
    batch_ind = []
    row_ind = []
    col_ind = []
    for i in range(distance_mat.shape[0]):
        dmat = distance_mat[i, :, :]
        if row_counts is not None:
            dmat = dmat[:row_counts[i], :]
        if col_counts is not None:
            dmat = dmat[:, :col_counts[i]]

        rind, cind = linear_sum_assignment(dmat.to('cpu').numpy())
        rind = list(rind)
        cind = list(cind)

        if len(rind) > 0:
            rind, cind = zip(*sorted(zip(rind, cind)))
            rind = list(rind)
            cind = list(cind)

        batch_ind += [i]*len(rind)
        row_ind += rind
        col_ind += cind

    return batch_ind, row_ind, col_ind


def collate_feats(b):
    return list(zip(*b))


def one_hot(inp, label_count):
    out = torch.zeros(label_count, inp.numel(), dtype=torch.uint8, device=inp.device)
    out[inp.view(-1), torch.arange(out.shape[1])] = 1
    out = out.view((label_count,) + inp.shape)
    return out


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, gt):
        smooth = 1e-6
        intersection = pred * gt
        union = pred + gt - intersection
        loss = (intersection / (union + smooth)).mean((1, 2, 3))
        loss = 1 - loss
        return loss


def sym_reflect_tree(tree):
    root_geo = tree.root.geo
    root_geo = torch.flip(root_geo, dims=(-1,))
    tree.root.geo = root_geo

    for i in range(len(tree.root.children)):
        child_geo = tree.root.children[i].geo
        child_geo = torch.flip(child_geo, dims=(3,))
        tree.root.children[i].geo = child_geo
    return tree


def mse_loss(pred, gt):
    mseLoss = nn.MSELoss()
    loss = mseLoss(pred, gt)
    avg_loss = loss.mean()

    return avg_loss


