#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
from torch.nn import Softmax
import torch.nn as nn

def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-10
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 


def im(outputs_test, gent=True):
    # print("outputs_test:", outputs_test.shape)             #outputs_test: torch.Size([16, 31])
    # print("outputs_test:", outputs_test)
    epsilon = 1e-10

    softmax_out = nn.Softmax(dim=1)(outputs_test)
    # print("softmax_out:", softmax_out.shape)              #softmax_out: torch.Size([16, 31])


    entropy_loss = torch.mean(entropy(softmax_out))
    # print("entropy_loss:", entropy_loss.shape)
    # print("entropy_loss:", entropy_loss)                #entropy_loss: tensor(4.8785, device='cuda:0', grad_fn=<MeanBackward0>)

    if gent:
        msoftmax = softmax_out.mean(dim=0)
        # print("msoftmax:", msoftmax.shape)             #msoftmax: torch.Size([31])
        # print("msoftmax:", msoftmax)

        gentropy_loss = torch.sum(-msoftmax * torch.log2(msoftmax + epsilon))
        # print("gentropy_loss:", gentropy_loss.shape)
        # print("gentropy_loss:", gentropy_loss)         #gentropy_loss: tensor(4.9461, device='cuda:0', grad_fn=<SumBackward0>)


        entropy_loss -= gentropy_loss
        # print("entropy_loss:", entropy_loss.shape)
        # print("entropy_loss:", entropy_loss)          #entropy_loss: tensor(-0.0652, device='cuda:0', grad_fn=<SubBackward0>)


    im_loss = entropy_loss * 1.0
    # print("im_loss:", im_loss.shape)
    # print("im_loss:", im_loss)                      #im_loss: tensor(-0.0652, device='cuda:0', grad_fn=<MulBackward0>)



    return im_loss


def adv(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(features.device)
    return torch.nn.BCELoss()(ad_out, dc_target)


def adv_local(features, ad_net, is_source=False, weights=None):
    ad_out = ad_net(features).squeeze(3)
    batch_size = ad_out.size(0)
    num_heads = ad_out.size(1)
    seq_len = ad_out.size(2)
    
    if is_source:
        label = torch.from_numpy(np.array([[[1]*seq_len]*num_heads] * batch_size)).float().to(features.device)
    else:
        label = torch.from_numpy(np.array([[[0]*seq_len]*num_heads] * batch_size)).float().to(features.device)

    return ad_out, torch.nn.BCELoss()(ad_out, label)
