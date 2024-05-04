# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2
from . import lmmd
from torch.nn.functional import normalize
import torch.nn.functional as F
from models.dynamic_conv import Dynamic_conv2d
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


class DynamicDWConv(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x_att, x_fea):
        # print("x1:", x_att.shape)                                                          #x1: torch.Size([16, 768, 16, 16])          torch.Size([16, 1536, 16, 16])
        b, c, h, w = x_att.shape
        weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x_att)))))
        # print("weight1:", weight.shape)                                                #weight1: torch.Size([16, 37632, 1, 1])          torch.Size([16, 75264, 1, 1])
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        # print("weight2:", weight.shape)                                                #weight2: torch.Size([12288, 1, 7, 7])           torch.Size([24576, 1, 7, 7])
        x_fea = F.conv2d(x_fea.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        # print("x2:", x_fea.shape)                                                          #x2: torch.Size([1, 12288, 16, 16])          torch.Size([1, 24576, 16, 16])
        x_fea = x_fea.view(b, c, x_fea.shape[-2], x_fea.shape[-1])
        # print("x3:", x_fea.shape)                                                          #x3: torch.Size([16, 768, 16, 16])           torch.Size([16, 1536, 16, 16])
        return x_fea




def conv1x1(in_planes, out_planes, stride=1):
    # return DynamicDWConv(in_planes, kernel_size=1, stride=1, padding=0, groups=in_planes)
    return Dynamic_conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Dynamic_conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False, dilation=dilation)


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Dynamic_conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, groups=groups, bias=False, dilation=dilation)

def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Dynamic_conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, groups=groups, bias=False, dilation=dilation)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        ######################################################
        # self.activation = nn.Sigmoid()
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.c = nn.Parameter(torch.ones(1) * 0)
        #######################################################


        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states_kv, hidden_states_q=None, domain=False):
        # print("hidden_states:", hidden_states.shape)     #hidden_states: torch.Size([16, 257, 768])
        ##################################################################
        # hidden_states_0 = hidden_states[:, 0]
        # print("hidden_states_0:", hidden_states_0.shape)      #hidden_states_0: torch.Size([16, 768])
        # hidden_states_1 = hidden_states[:, 1:]
        # print("hidden_states_1:", hidden_states_1.shape)       #hidden_states_1: torch.Size([16, 256, 768])
        # hidden_states_1 = rearrange(hidden_states_1, 'b hw c -> b c hw')
        # channel_mean_s = hidden_states_1.mean(-1, keepdim=True)
        # channel_var_s = hidden_states_1.var(-1, keepdim=True)
        # hidden_states_1 = (hidden_states_1 - channel_mean_s) / (channel_var_s + 1e-5).sqrt()
        # hidden_states_1 = rearrange(hidden_states_1, 'b c hw -> b hw c')
        # hidden_states_0 = hidden_states_0.unsqueeze(1)
        # hidden_states = torch.cat((hidden_states_0, hidden_states_1), dim=1)



        ##################################################################
        if domain:
            mixed_query_layer = self.query(hidden_states_q)
            mixed_key_layer = self.key(hidden_states_kv)
            mixed_value_layer = self.value(hidden_states_kv)
        else:
            mixed_query_layer = self.query(hidden_states_kv)
            mixed_key_layer = self.key(hidden_states_kv)
            mixed_value_layer = self.value(hidden_states_kv)

        ############################22.54######################################
        # mixed_value_layer_0 = mixed_value_layer[:, 0]
        # mixed_value_layer_1 = mixed_value_layer[:, 1:]
        # mixed_value_layer_1 = rearrange(mixed_value_layer_1, 'b hw c -> b c hw')
        # channel_mean_s = mixed_value_layer_1.mean(-1, keepdim=True)
        # channel_var_s = mixed_value_layer_1.var(-1, keepdim=True)
        # mixed_value_layer_1 = (mixed_value_layer_1 - channel_mean_s) / (channel_var_s + 1e-5).sqrt()
        # mixed_value_layer_1 = rearrange(mixed_value_layer_1, 'b c hw -> b hw c')
        # mixed_value_layer_0 = mixed_value_layer_0.unsqueeze(1)
        # mixed_value_layer = torch.cat((mixed_value_layer_0, mixed_value_layer_1), dim=1)
        ##################################################################

        ##############################79.162####################################
        # mixed_value_layer_0 = mixed_value_layer[:, 0]
        # print("mixed_value_layer_0:", mixed_value_layer_0.shape)      #mixed_value_layer_0: torch.Size([16, 768])
        # mixed_value_layer_0 = mixed_value_layer_0.unsqueeze(1)
        # print("mixed_value_layer_0:", mixed_value_layer_0.shape)      #mixed_value_layer_0: torch.Size([16, 1, 768])
        # mixed_value_layer_1 = mixed_value_layer[:, 1:]
        # print("mixed_value_layer_1:", mixed_value_layer_1.shape)      #mixed_value_layer_1: torch.Size([16, 256, 768])
        # mixed_value_layer_1 = rearrange(mixed_value_layer_1, 'b (h w) c -> b c h w', h=16, w=16)
        # print("mixed_value_layer_1:", mixed_value_layer_1.shape)      #mixed_value_layer_1: torch.Size([16, 768, 16, 16])
        # mixed_value_layer_1_avg = self.avgpool(mixed_value_layer_1)
        # print("mixed_value_layer_1_avg:", mixed_value_layer_1_avg.shape)      #mixed_value_layer_1: torch.Size([16, 768, 1, 1])
        # gate_t = self.activation(mixed_value_layer_1_avg)
        # print("gate_t:", gate_t.shape)                                #gate_t: torch.Size([16, 768, 1, 1])
        # mixed_value_layer_1 = mixed_value_layer_1 * gate_t  # x_s: torch.Size([16, 768, 1, 1])
        # print("mixed_value_layer_1:", mixed_value_layer_1.shape)          #mixed_value_layer_1: torch.Size([16, 768, 16, 16])

        # mixed_value_layer_0 = rearrange(mixed_value_layer_0, 'b c hw -> b hw c')
        # mixed_value_layer_1 = rearrange(mixed_value_layer_1, 'b c h w -> b (h w) c')
        # print("mixed_value_layer_1:", mixed_value_layer_1.shape)          #mixed_value_layer: torch.Size([16, 256, 768])
        # mixed_value_layer = torch.cat((mixed_value_layer_0, mixed_value_layer_1), dim=1)
        # print("mixed_value_layer:", mixed_value_layer.shape)          #mixed_value_layer: torch.Size([16, 257, 768])

        # mixed_value_layer_0 = mixed_value_layer[:, 0]
        # mixed_value_layer_0_un = mixed_value_layer[:, 0].unsqueeze(1)
        # mixed_value_layer_0_un = rearrange(mixed_value_layer_0_un, 'b (h w) c -> b c h w', h=1, w=1)

        # mixed_value_layer_1 = mixed_value_layer[:, 1:]
        # mixed_value_layer_1 = rearrange(mixed_value_layer_1, 'b (h w) c -> b c h w', h=16, w=16)
        # norm
        # attn = self.norm(mixed_value_layer_0)
        # attn = self.gaussian(attn, 2)
        # attn = self.gaussian(attn, 3 * torch.sigmoid(self.c) + 1)
        # attn = attn.unsqueeze(-1).unsqueeze(-1)
        # mixed_value_layer_0_un = mixed_value_layer_0_un * attn
        # mixed_value_layer_1 = mixed_value_layer_1 * attn
        # mixed_value_layer_0_un = rearrange(mixed_value_layer_0_un, 'b c h w -> b (h w) c', h=1, w=1)
        # mixed_value_layer_1 = rearrange(mixed_value_layer_1, 'b c h w -> b (h w) c', h=16, w=16)
        # mixed_value_layer = torch.cat((mixed_value_layer_0_un, mixed_value_layer_1), dim=1)



        #################################################################


        ##################################################################
        # mixed_query_layer_0 = mixed_query_layer[:, 0]
        # mixed_query_layer_1 = mixed_query_layer[:, 1:]
        # mixed_query_layer_1 = rearrange(mixed_query_layer_1, 'b hw c -> b c hw')
        # channel_mean_s = mixed_query_layer_1.mean(-1, keepdim=True)
        # channel_var_s = mixed_query_layer_1.var(-1, keepdim=True)
        # mixed_query_layer_1 = (mixed_query_layer_1 - channel_mean_s) / (channel_var_s + 1e-5).sqrt()
        # mixed_query_layer_1 = rearrange(mixed_query_layer_1, 'b c hw -> b hw c')
        # mixed_query_layer_0 = mixed_query_layer_0.unsqueeze(1)
        # mixed_query_layer = torch.cat((mixed_query_layer_0, mixed_query_layer_1), dim=1)
        #
        # mixed_key_layer_0 = mixed_key_layer[:, 0]
        # mixed_key_layer_1 = mixed_key_layer[:, 1:]
        # mixed_key_layer_1 = rearrange(mixed_key_layer_1, 'b hw c -> b c hw')
        # channel_mean_s = mixed_key_layer_1.mean(-1, keepdim=True)
        # channel_var_s = mixed_key_layer_1.var(-1, keepdim=True)
        # mixed_key_layer_1 = (mixed_key_layer_1 - channel_mean_s) / (channel_var_s + 1e-5).sqrt()
        # mixed_key_layer_1 = rearrange(mixed_key_layer_1, 'b c hw -> b hw c')
        # mixed_key_layer_0 = mixed_key_layer_0.unsqueeze(1)
        # mixed_key_layer = torch.cat((mixed_key_layer_0, mixed_key_layer_1), dim=1)

        ##################################################################



        # print("mixed_query_layer:", mixed_query_layer.shape)     #mixed_query_layer: torch.Size([16, 257, 768])
        # print("mixed_key_layer:", mixed_key_layer.shape)
        # print("mixed_value_layer:", mixed_value_layer.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # print("query_layer:", query_layer.shape)                 #query_layer: torch.Size([16, 12, 257, 64])
        # print("key_layer:", key_layer.shape)
        # print("value_layer:", value_layer.shape)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # print("attention_scores1:", attention_scores.shape)        #attention_scores1: torch.Size([16, 12, 257, 257])
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print("attention_scores2:", attention_scores.shape)  #attention_scores2: torch.Size([16, 12, 257, 257])


        attention_probs = self.softmax(attention_scores)
        # print("attention_probs1:", attention_probs.shape)  #attention_probs1: torch.Size([16, 12, 257, 257])
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        # print("attention_probs2:", attention_probs.shape)  #attention_probs2: torch.Size([16, 12, 257, 257])

        context_layer = torch.matmul(attention_probs, value_layer)
        # print("context_layer:", context_layer.shape)                 #    context_layer: torch.Size([16, 12, 257, 64])


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # print("context_layer:", context_layer.shape)                 #context_layer: torch.Size([16, 257, 12, 64])


        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # print("new_context_layer_shape:", new_context_layer_shape.shape)



        context_layer = context_layer.view(*new_context_layer_shape)
        # print("context_layer:", context_layer.shape)           #context_layer: torch.Size([16, 257, 768])


        attention_output = self.out(context_layer)
        # print("attention_output:", attention_output.shape)     #attention_output: torch.Size([16, 257, 768])


        attention_output = self.proj_dropout(attention_output)
        # print("attention_output:", attention_output.shape)     #attention_output: torch.Size([16, 257, 768])



        # print("attention_output:", attention_output.shape)   #attention_output: torch.Size([16, 257, 768])
        ################################82.286##################################
        # attention_output_0 = attention_output[:, 0]
        # attention_output_0 = attention_output_0.unsqueeze(1)
        # attention_output_1 = attention_output[:, 1:]
        # attention_output_0 = rearrange(attention_output_0, 'b hw c -> b c hw')
        # gate_t = self.activation(attention_output_0)  # gate_s: torch.Size([16, 768, 1, 1])
        # attention_output_0 = attention_output_0 * gate_t  # x_s: torch.Size([16, 768, 1, 1])
        # attention_output_0 = rearrange(attention_output_0, 'b c hw -> b hw c')
        # attention_output = torch.cat((attention_output_0, attention_output_1), dim=1)
        #################################################################


        #################################################################
        # attention_output_0 = attention_output[:, 0]
        # attention_output_un = attention_output[:, 0].unsqueeze(1)
        # attention_output_un = rearrange(attention_output_un, 'b (h w) c -> b c h w', h=1, w=1)
        #
        # attention_output_1 = attention_output[:, 1:]
        # attention_output_1 = rearrange(attention_output_1, 'b (h w) c -> b c h w', h=16, w=16)
        # norm
        # attn = self.norm(attention_output_0)
        # attn = self.gaussian(attn, 2)
        # attn = attn.unsqueeze(-1).unsqueeze(-1)
        # attention_output_un = attention_output_un * attn
        # attention_output_1 = attention_output_1 * attn
        # attention_output_un = rearrange(attention_output_un, 'b c h w -> b (h w) c', h=1, w=1)
        # attention_output_1 = rearrange(attention_output_1, 'b c h w -> b (h w) c', h=16, w=16)
        # attention_output = torch.cat((attention_output_un, attention_output_1), dim=1)


        ##################################################################


        return attention_output, weights



    # @staticmethod
    # def norm(x):
    #     mean = x.mean(dim=-1, keepdim=True).expand_as(x)
    #     std = x.std(dim=-1, keepdim=True).expand_as(x)
    #     rst = (x - mean) / std
    #     return rst
    #
    # @staticmethod
    # def gaussian(x, c):
    #     return torch.exp(-(x ** 2) / (2 * c))

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        ######################################################
        self.c = nn.Parameter(torch.ones(1) * 0)
        #######################################################

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        # print("x1:", x.shape)   #x1: torch.Size([16, 257, 768])
        x = self.fc1(x)
        # print("x2:", x.shape)   #x2: torch.Size([16, 257, 3072])
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # print("x5:", x.shape)  #x5: torch.Size([16, 257, 768])
        x = self.dropout(x)

        ####################################Gassusion Channel Attention########################################
        x_0 = x[:, 0]
        x_0_un = x[:, 0].unsqueeze(1)
        x_0_un = rearrange(x_0_un, 'b (h w) c -> b c h w', h=1, w=1)
        x_1 = x[:, 1:]
        x_1_un = rearrange(x_1, 'b (h w) c -> b c h w', h=16, w=16)
        #
        # norm
        attn = self.norm(x_0)
        attn = self.gaussian(attn, 3 * torch.sigmoid(self.c) + 1)
        attn = attn.unsqueeze(-1).unsqueeze(-1)
        x_0_un = x_0_un * attn
        x_1_un = x_1_un * attn
        #
        x_0_un = rearrange(x_0_un, 'b c h w -> b (h w) c', h=1, w=1)
        x_1_un = rearrange(x_1_un, 'b c h w -> b (h w) c', h=16, w=16)
        x = torch.cat((x_0_un, x_1_un), dim=1)
        ############################################################################

        return x

    @staticmethod
    def norm(x):
        mean = x.mean(dim=-1, keepdim=True).expand_as(x)
        std = x.std(dim=-1, keepdim=True).expand_as(x)
        rst = (x - mean) / std
        return rst
    #
    @staticmethod
    def gaussian(x, c):
        return torch.exp(-(x ** 2) / (2 * c))


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        ######################################################
        # self.activation = nn.Sigmoid()
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        #######################################################
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        # print("x1:", x.shape)    #x1: torch.Size([16, 3, 256, 256])
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        # print("x2:", x.shape)    #x2: torch.Size([16, 768, 16, 16])

        x = x.flatten(2)

        # print("x3:", x.shape)    #x3: torch.Size([16, 768, 256])
        x = x.transpose(-1, -2)
        # print("x4:", x.shape)    #x4: torch.Size([16, 256, 768])
        # print("cls_tokens:", cls_tokens.shape)    #cls_tokens: torch.Size([16, 1, 768])
        x = torch.cat((cls_tokens, x), dim=1)
        # print("x5:", x.shape)    #x5: torch.Size([16, 257, 768])
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # self.attention_norm_q = LayerNorm(config.hidden_size, eps=1e-6)
        ############################################
        # self.bn1 = nn.BatchNorm2d(768)
        ######################################
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x, x_q=None, domain=False):
        h = x
        # print("x_pre:", x.shape)        #   x_pre: torch.Size([16, 257, 768])
        x = self.attention_norm(x)

        if domain:
            x_q = self.attention_norm(x_q)

        x, weights = self.attn(x, x_q, domain)
        x = x + h

        h = x
        # print("x2_pre:", x.shape)        #x2_pre: torch.Size([16, 257, 768])
        x = self.ffn_norm(x)

        # print("x2_post:", x.shape)        #x2_post: torch.Size([16, 257, 768])
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states_kv, hidden_states_q=None, domain=False):
        attn_weights = []
        hidden_states_stack = torch.zeros([12, 8, 257, 768])

        # print("hidden_states_kv11:", hidden_states_kv11.shape)

        ########################## stack cross ###################################
        if domain is True:
            for i, layer_block in enumerate(self.layer):
                hidden_states_kv, weights = layer_block(hidden_states_kv, hidden_states_q[i], domain)
                # hidden_states_kv, weights = layer_block(hidden_states_kv, hidden_states_kv, domain)
                if self.vis:
                    attn_weights.append(weights)
        else:
            for i, layer_block in enumerate(self.layer):
                if i == 0:
                    hidden_states_stack = hidden_states_kv.unsqueeze(0)
                    hidden_states_kv, weights = layer_block(hidden_states_kv, hidden_states_q, domain)
                    # print("i:", i)
                else:
                    hidden_states_stack = torch.cat((hidden_states_stack, hidden_states_kv.unsqueeze(0)), 0)
                    hidden_states_kv, weights = layer_block(hidden_states_kv, hidden_states_q, domain)
                    # print("i:", i)
                if self.vis:
                    attn_weights.append(weights)

        ########################################################################

        encoded = self.encoder_norm(hidden_states_kv)


        return encoded, attn_weights, hidden_states_stack


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, domain):
        # print("input_ids:", input_ids.shape)                       #input_ids: torch.Size([16, 3, 256, 256])
        embedding_output = self.embeddings(input_ids)
        # print("embedding_output:", embedding_output.shape)         #embedding_output: torch.Size([16, 257, 768])
        encoded, attn_weights, hidden_states_kv = self.encoder(embedding_output, domain)
        # print("encoded:", encoded.shape)                           #encoded: torch.Size([16, 257, 768])
        return encoded, attn_weights, hidden_states_kv, embedding_output


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=31, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)


        self.head1 = conv1x1(768, 768)
        # self.head1 = nn.Conv2d(768, 768, kernel_size=1, padding=0, stride=1, bias=False)
        ###########################################
        self.layer_norm1 = LayerNorm(768, eps=1e-6)
        ################################################


        self.cluster_projector = nn.Sequential(
            # nn.Linear(768, 768),
            nn.Linear(1536, 256),
            # nn.Linear(256, 256),
            # nn.Conv2d(768, 256, kernel_size=1, padding=0, stride=1, bias=False),

            # conv1x1(768, 256),
            # conv1x1(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.Linear(256, 256)
        )

        # self.head2 = conv1x1(256, 256)
        # self.head2 = nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False)
        self.head2 = Linear(256, 256)
        self.head3 = Linear(256, num_classes)
        # self.head3 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, stride=1, bias=False)
        # self.head3 = conv1x1(256, num_classes)

        self.lmmd_loss = lmmd.LMMD_loss(class_num=num_classes)

    # def random_masking(self, x, y, mask_ratio):
    #     N, L, D = x.shape  # batch, length, dim
    #     print("x:", x.shape)         #([1, 256, 768])
        # len_keep = int(L * (1 - mask_ratio))
        # print("len_keep:", len_keep)     #len_keep: 128
        # noise = torch.rand(N, L)  # noise in [0, 1]
        # noise = noise.cuda()
        #
        # mate = x
        # y = 0
        # mate = x+y
        #
        # return mate


    def random_masking_0(self, x, y, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        ###################################xxxx###############################################
        N, L, D = x.shape  # batch, length, dim
        # print("x:", x.shape)         #([1, 256, 768])
        len_keep = int(L * (1 - mask_ratio))
        # print("len_keep:", len_keep)     #len_keep: 128

        # 生成一组随机数
        # noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise = torch.rand(N, L)  # noise in [0, 1]
        noise = noise.cuda()
        # print("noise:", noise.shape)      #noise: torch.Size([1, 256])  (0~1)
        # print("noise:", noise)

        # sort noise for each sample
        # 将随机数组从小到大排序，并得到排序的index(相当于shuffle)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print("ids_shuffle:", ids_shuffle.shape)    #ids_shuffle: torch.Size([1, 256])  (0-255)
        # print("ids_shuffle:", ids_shuffle)


        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # print("ids_keep:", ids_keep.shape)       #ids_keep: torch.Size([1, 128]
        # print("ids_keep:", ids_keep)

        # 从对应index取值
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # print("x_masked:", x_masked.shape)             #x_masked: torch.Size([1, 128, 768])
        # print("x_masked:", x_masked)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        # print("mask1:", mask)
        # print("mask:", mask.shape)
        mask[:, :len_keep] = 0
        # print("mask2:", mask)
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_shuffle).bool()
        # print("mask3:", mask.shape)
        # print("mask3:", mask)
        mask = mask.unsqueeze(-1).repeat(1, 1, D)
        # print("mask4:", mask.shape)
        # print("mask4:", mask)
        # mask = = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).bool()

        # print("x1:", x.shape)
        # print("x1:", x)
        x = x.masked_fill(mask, 0)
        # x = x
        # print("x2:", x.shape)
        # print("x2:", x)
        #############################################################################################


        #############################################################################################

        ###################################YYYY###############################################
        N_t, L_t, D_t = y.shape  # batch, length, dim
        # len_keep_t = int(L * (1 - mask_ratio))
        # 生成一组随机数
        # noise_t = torch.rand(N_t, L_t, device=y.device)  # noise in [0, 1]

        # sort noise for each sample
        # 将随机数组从小到大排序，并得到排序的index(相当于shuffle)
        # ids_shuffle_t = torch.argsort(noise_t, dim=1)  # ascend: small is keep, large is remove
        # 给随机排序的index再从小到大排序，等价于reshuffle
        # ids_restore_t = torch.argsort(ids_shuffle_t, dim=1)

        # keep the first subset
        ids_keep_t = ids_shuffle[:, len_keep:]
        # print("ids_keep_t:", ids_keep_t.shape)       #ids_keep: torch.Size([1, 128]
        # print("ids_keep_t:", ids_keep_t)
        # 从对应index取值
        y_masked = torch.gather(y, dim=1, index=ids_keep_t.unsqueeze(-1).repeat(1, 1, D_t))
        # print("y_masked:", y_masked.shape)             #x_masked: torch.Size([1, 128, 768])
        # print("y_masked:", y_masked)

        y = y.masked_fill(~mask, 0)
        # print("y:", y.shape)
        # print("y:", y)
        #############################################################################################
        # masked_cat = torch.cat((x_masked, y_masked), dim=1)
        masked_cat = x + y
        # masked_cat = x
        # print("masked_cat:", masked_cat.shape)             #x_masked: torch.Size([1, 128, 768])
        # print("masked_cat:", masked_cat)

        return masked_cat, noise

    def noise_value(self, x):
        # print("x:", x.shape)
        _, N, L, _ = x.shape  # batch, length, dim
        noise = torch.rand(N, L)  # noise in [0, 1]
        noise = noise.cuda()
        return noise

    def random_masking_1(self, x, y, noise, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        ###################################xxxx###############################################
        N, L, D = x.shape  # batch, length, dim
        # print("x:", x.shape)         #([1, 256, 768])
        len_keep = int(L * (1 - mask_ratio))
        # print("len_keep:", len_keep)     #len_keep: 128

        # 生成一组随机数
        # noise = noise  # noise in [0, 1]
        # print("noise:", noise.shape)      #noise: torch.Size([1, 256])  (0~1)
        # print("noise:", noise)

        # sort noise for each sample
        # 将随机数组从小到大排序，并得到排序的index(相当于shuffle)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print("ids_shuffle:", ids_shuffle.shape)    #ids_shuffle: torch.Size([1, 256])  (0-255)
        # print("ids_shuffle:", ids_shuffle)


        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # print("ids_keep:", ids_keep.shape)       #ids_keep: torch.Size([1, 128]
        # print("ids_keep:", ids_keep)

        # 从对应index取值
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # print("x_masked:", x_masked.shape)             #x_masked: torch.Size([1, 128, 768])
        # print("x_masked:", x_masked)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        # print("mask1:", mask)
        # print("mask:", mask.shape)
        mask[:, :len_keep] = 0
        # print("mask2:", mask)
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_shuffle).bool()
        # print("mask3:", mask.shape)
        # print("mask3:", mask)
        mask = mask.unsqueeze(-1).repeat(1, 1, D)
        # print("mask4:", mask.shape)
        # print("mask4:", mask)
        # mask = = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).bool()
        # print("x1:", x.shape)
        # print("x1:", x)



        # x = x.masked_fill(mask, 0)
        # print("x:", x.shape)
        # print("x:", x)
        # print("x2:", x.shape)
        # print("x2:", x)

        #############################################################################################

        ###################################YYYY###############################################
        N_t, L_t, D_t = y.shape  # batch, length, dim
        # len_keep_t = int(L * (1 - mask_ratio))
        # 生成一组随机数
        # noise_t = torch.rand(N_t, L_t, device=y.device)  # noise in [0, 1]

        # sort noise for each sample
        # 将随机数组从小到大排序，并得到排序的index(相当于shuffle)
        # ids_shuffle_t = torch.argsort(noise_t, dim=1)  # ascend: small is keep, large is remove
        # 给随机排序的index再从小到大排序，等价于reshuffle
        # ids_restore_t = torch.argsort(ids_shuffle_t, dim=1)

        # keep the first subset
        ids_keep_t = ids_shuffle[:, len_keep:]
        # print("ids_keep_t:", ids_keep_t.shape)       #ids_keep: torch.Size([1, 128]
        # print("ids_keep_t:", ids_keep_t)
        # 从对应index取值
        y_masked = torch.gather(y, dim=1, index=ids_keep_t.unsqueeze(-1).repeat(1, 1, D_t))
        # print("y_masked:", y_masked.shape)             #x_masked: torch.Size([1, 128, 768])
        # print("y_masked:", y_masked)

        y = y.masked_fill(~mask, 0)

        #############################################################################################
        # masked_cat = torch.cat((x_masked, y_masked), dim=1)
        # masked_cat = x + y
        masked_cat = x
        # print("masked_cat:", masked_cat.shape)             #x_masked: torch.Size([1, 128, 768])
        # print("masked_cat:", masked_cat)

        # print("masked_cat:", masked_cat.shape)             #x_masked: torch.Size([1, 128, 768])
        # print("masked_cat:", masked_cat)
        #
        return masked_cat


    def forward(self, x_s, x_t=None, labels=None, eps=1e-5):

        if x_t is not None:

            x_s, attn_weights_s, hidden_states_s, _ = self.transformer(x_s, domain=False)
            out_s1_cls_token = x_s[:, 0]
            out_s1 = x_s[:, 1:]
            out_s1 = out_s1.mean(dim=1)

            x_t, _, hidden_states_t, embedding_output = self.transformer(x_t, domain=False)
            out_t1_cls_token = x_t[:, 0]
            out_t1 = x_t[:, 1:]
            out_t1 = out_t1.mean(dim=1)

            ##############################84.310############################################
            out_s1_cls_token = out_s1_cls_token.unsqueeze(1)
            out_s1_cls_token_style = rearrange(out_s1_cls_token, 'b (h w) c -> b c h w', h=1, w=1)
            out_s1_cls_token = self.head1(out_s1_cls_token_style, out_s1_cls_token_style)    #x1=style x2=content
            out_s1_cls_token = rearrange(out_s1_cls_token, 'b c h w-> b (h w) c', h=1, w=1)
            out_s1_cls_token = self.layer_norm1(out_s1_cls_token)
            out_s1_cls_token = out_s1_cls_token.squeeze(1)

            out_s1 = out_s1.unsqueeze(1)
            out_s1_style = rearrange(out_s1, 'b (h w) c -> b c h w', h=1, w=1)
            out_s1 = self.head1(out_s1_style, out_s1_style)
            out_s1 = rearrange(out_s1, 'b c h w-> b (h w) c', h=1, w=1)
            out_s1 = self.layer_norm1(out_s1)
            out_s1 = out_s1.squeeze(1)

            out_s1_cls_token = torch.cat((out_s1_cls_token, out_s1), dim=1)
            ##########################################################################


            x_s_1 = self.cluster_projector(out_s1_cls_token)
            x_s_1 = self.head2(x_s_1)
            logits_s = self.head3(x_s_1)


            ##############################84.310###############cat 84.594#############################
            out_t1_cls_token = out_t1_cls_token.unsqueeze(1)
            out_t1_cls_token_style = rearrange(out_t1_cls_token, 'b (h w) c -> b c h w', h=1, w=1)
            out_t1_cls_token = self.head1(out_t1_cls_token_style, out_t1_cls_token_style)
            out_t1_cls_token = rearrange(out_t1_cls_token, 'b c h w-> b (h w) c', h=1, w=1)
            out_t1_cls_token = self.layer_norm1(out_t1_cls_token)
            out_t1_cls_token = out_t1_cls_token.squeeze(1)

            out_t1 = out_t1.unsqueeze(1)
            out_t1_style = rearrange(out_t1, 'b (h w) c -> b c h w', h=1, w=1)
            out_t1 = self.head1(out_t1_style, out_t1_style)
            out_t1 = rearrange(out_t1, 'b c h w-> b (h w) c', h=1, w=1)
            out_t1 = self.layer_norm1(out_t1)
            out_t1 = out_t1.squeeze(1)

            out_t1_cls_token = torch.cat((out_t1_cls_token, out_t1), dim=1)
            ##########################################################################

            x_t_1 = self.cluster_projector(out_t1_cls_token)

            x_t_1 = self.head2(x_t_1)

            logits_t = self.head3(x_t_1)

            ########################################cross begin####################################
            # s_index_adjust, t_index_adjust = self.lmmd_loss.cal_adjust(labels, torch.nn.functional.softmax(logits_t,dim=1))
            ##########################################

            # print("s_index_adjust:", s_index_adjust)
            # print("s_index_adjust:", len(s_index_adjust))
            # print("t_index_adjust:", t_index_adjust)

            # noise = self.noise_value(hidden_states_s)
            # noise = self.noise_value(hidden_states_s[::, s_index_adjust, 1:, ::])


            # if len(s_index_adjust) != 0:
            #     for i in range(12):
            #         print("i:", i)
            #         hidden_states_t[i, t_index_adjust, ::, ::] = hidden_states_s[i, s_index_adjust, ::, ::]
                    # hidden_states_t[i, t_index_adjust, 0, ::] = hidden_states_s[i, s_index_adjust, 0, ::]
                    # hidden_states_t[i, t_index_adjust, 1:, ::] = hidden_states_s[i, s_index_adjust, 1:, ::]
                    # hidden_states_t[i, t_index_adjust, 1:, ::] = self.random_masking_1(
                    #     hidden_states_s[i, s_index_adjust, 1:, ::], hidden_states_t[i, t_index_adjust, 1:, ::], noise,
                    #     mask_ratio=0)
                    # print("hidden_states_t[i, t_index_adjust, ::, ::]:", hidden_states_t[i, t_index_adjust, ::, ::].shape)
                    # print("hidden_states_t[i, t_index_adjust, ::, ::]:", hidden_states_t[i, t_index_adjust, ::, ::])
            #

            # print("hidden_states_s:", hidden_states_s.shape)
            # print("hidden_states_s[s_index_adjust, 1:, ::]:", hidden_states_s[::, s_index_adjust, 1:, ::].shape)
            # noise = self.noise_value(hidden_states_s[::, s_index_adjust, 1:, ::])
            # print("noise:", noise.shape)

            # if len(s_index_adjust) != 0:
            #     for i in range(12):
                    # print("i:", i)
                    # hidden_states_t[i, t_index_adjust, 0, ::] = hidden_states_s[i, s_index_adjust, 0, ::]
                    # hidden_states_t[i, t_index_adjust, 1:, ::] = hidden_states_s[i, s_index_adjust, 1:, ::]
                    # hidden_states_t[i, t_index_adjust, 1:, ::] = self.random_masking_1(hidden_states_s[i, s_index_adjust, 1:, ::], hidden_states_t[i, t_index_adjust, 1:, ::], noise, mask_ratio=0)
                    # print("hidden_states_t[i, t_index_adjust, ::, ::]:", hidden_states_t[i, t_index_adjust, ::, ::].shape)
                    # print("hidden_states_t[i, t_index_adjust, ::, ::]:", hidden_states_t[i, t_index_adjust, ::, ::])






            # if len(s_index_adjust) != 0:
            #     for i in range(12):
                    # print("i:", i)
                    # if i == 0:
                    #     hidden_states_t[i, t_index_adjust, 1:, ::], noise = self.random_masking_0(
                    #         hidden_states_s[i, s_index_adjust, 1:, ::], hidden_states_t[i, t_index_adjust, 1:, ::],
                    #         mask_ratio=0)
                    #     hidden_states_t[i, t_index_adjust, 0, ::] = hidden_states_s[i, s_index_adjust, 0, ::]
                        # print("hidden_states_t[i, t_index_adjust, ::, ::]:", hidden_states_t[i, t_index_adjust, ::, ::].shape)
                        # print("hidden_states_t[i, t_index_adjust, ::, ::]:", hidden_states_t[i, t_index_adjust, ::, ::])
                    # else:
                    #     hidden_states_t[i, t_index_adjust, 1:, ::] = self.random_masking_1(
                    #         hidden_states_s[i, s_index_adjust, 1:, ::], hidden_states_t[i, t_index_adjust, 1:, ::], noise,
                    #         mask_ratio=0)
                    #     hidden_states_t[i, t_index_adjust, 0, ::] = hidden_states_s[i, s_index_adjust, 0, ::]
                        # print("hidden_states_t[i, t_index_adjust, ::, ::]:", hidden_states_t[i, t_index_adjust, ::, ::].shape)
                        # print("hidden_states_t[i, t_index_adjust, ::, ::]:", hidden_states_t[i, t_index_adjust, ::, ::])

            # x_t_q = hidden_states_t

            # x_t_cross, _, _ = self.transformer.encoder(embedding_output, x_t_q, domain=True)

            #########################################################
            # out_t2_cls_token = x_t_cross[:, 0]
            # out_t2 = x_t_cross[:, 1:]
            # out_t2 = out_t2.mean(dim=1)

            # out_t2_cls_token = out_t2_cls_token.unsqueeze(1)
            # out_t2_cls_token_style = rearrange(out_t2_cls_token, 'b (h w) c -> b c h w', h=1, w=1)
            # out_t2_cls_token = self.head1(out_t2_cls_token_style, out_t2_cls_token_style)
            # out_t2_cls_token = rearrange(out_t2_cls_token, 'b c h w-> b (h w) c', h=1, w=1)
            # out_t2_cls_token = self.layer_norm1(out_t2_cls_token)
            # out_t2_cls_token = out_t2_cls_token.squeeze(1)
            #
            # out_t2 = out_t2.unsqueeze(1)
            # out_t2_style = rearrange(out_t2, 'b (h w) c -> b c h w', h=1, w=1)
            # out_t2 = self.head1(out_t2_style, out_t2_style)
            # out_t2 = rearrange(out_t2, 'b c h w-> b (h w) c', h=1, w=1)
            # out_t2 = self.layer_norm1(out_t2)
            # out_t2 = out_t2.squeeze(1)
            #
            # out_t2_cls_token = torch.cat((out_t2_cls_token, out_t2), dim=1)
            ##########################################################################

            # x_t_2 = self.cluster_projector(out_t2_cls_token)
            #
            # x_t_2 = self.head2(x_t_2)
            #
            # logits_t_2 = self.head3(x_t_2)
            # s_index_adjust_2, t_index_adjust_2 = self.lmmd_loss.cal_adjust(labels, torch.nn.functional.softmax(logits_t_2, dim=1))
            ########################################cross end####################################


            # loss_lmmd = self.lmmd_loss.get_loss(x_s_1, x_t_1, labels, torch.nn.functional.softmax(logits_t, dim=1))

            # return logits_s, logits_t, loss_lmmd

            #######################################cross begin################################################
            loss_lmmd1 = self.lmmd_loss.get_loss(x_s_1, x_t_1, labels, torch.nn.functional.softmax(logits_t, dim=1))
            # loss_lmmd2 = self.lmmd_loss.get_loss(x_s_1, x_t_2, labels, torch.nn.functional.softmax(logits_t_2, dim=1))
            # return logits_s, logits_t, logits_t_2, loss_lmmd1, loss_lmmd2
            return logits_s, logits_t, loss_lmmd1
            #######################################cross end################################################


            # return logits_s, loss_lmmd
        else:

            # print("x_s_pre:", x_s.shape)      # x_s_pre: torch.Size([16, 3, 256, 256])
            x_s, attn_weights_s, hidden_states_s, em = self.transformer(x_s, domain=False)

            # print("x_s_post:", x_s.shape)     # x_s_post: torch.Size([16, 257, 768])
            out_s1_cls_token = x_s[:, 0]
            out_s1 = x_s[:, 1:]

            ##################################################################
            out_s1 = out_s1.mean(dim=1)
            #########################################################################
            ######################################################################
            out_s1_cls_token = out_s1_cls_token.unsqueeze(1)
            out_s1_cls_token = rearrange(out_s1_cls_token, 'b (h w) c -> b c h w', h=1, w=1)
            out_s1_cls_token = self.head1(out_s1_cls_token, out_s1_cls_token)
            out_s1_cls_token = rearrange(out_s1_cls_token, 'b c h w-> b (h w) c', h=1, w=1)
            out_s1_cls_token = self.layer_norm1(out_s1_cls_token)
            out_s1_cls_token = out_s1_cls_token.squeeze(1)

            out_s1 = out_s1.unsqueeze(1)
            out_s1 = rearrange(out_s1, 'b (h w) c -> b c h w', h=1, w=1)
            out_s1 = self.head1(out_s1, out_s1)
            out_s1 = rearrange(out_s1, 'b c h w-> b (h w) c', h=1, w=1)
            out_s1 = self.layer_norm1(out_s1)
            out_s1 = out_s1.squeeze(1)

            out_s1_cls_token = torch.cat((out_s1_cls_token, out_s1), dim=1)
            ######################################################################

            x_s_1 = self.cluster_projector(out_s1_cls_token)

            x_s_1 = self.head2(x_s_1)

            logits_s = self.head3(x_s_1)


            # logits_s = rearrange(logits_s, 'b c h w-> b (h w) c', h=1, w=1)
            # logits_s = logits_s.squeeze(1)


            return logits_s


    def load_from(self, weights):
        with torch.no_grad():
            # if self.zero_head:
            #     nn.init.zeros_(self.head.weight)
            #     nn.init.zeros_(self.head.bias)
            # else:
            #     self.head.weight.copy_(np2th(weights["head/kernel"]).t())
            #     self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
