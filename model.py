# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math
import argparse
import fit
# from shift import Shift
import numpy as np

paris = {
    'ntu/cv': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu/cs': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu120/xsetup': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu120/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
}


class SGN(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias=True):
        super(SGN, self).__init__()

        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg  # 20

        self.case = args.case
        num_joint = 25
        bs = args.batch_size
        if args.train:
            self.spa = self.one_hot(bs, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(bs, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()
        else:
            self.spa = self.one_hot(32 * 5, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(32 * 5, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, 64 * 4, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(3, 64, norm=True, bias=bias)
        self.dif_embed = embed(3, 64, norm=True, bias=bias)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        # self.compute_g2 = compute_g_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        # self.compute_g3 = compute_g_spa(self.dim1, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)

        # self.residual = cnn1x1(self.dim1 // 2, self.dim1)
        # self.bn_residual = nn.BatchNorm2d(self.dim1)

        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)

        self.fc = nn.Linear(self.dim1 * 2, num_classes)

        # attention
        # self.tem_attention = tem_attention(self.dim1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn1.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn1.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn1.weight, 0)

    def forward(self, input):

        # Dynamic Representation
        # 64, 20, 75
        bs, step, dim = input.size()
        # print(input.size())
        num_joints = dim // 3
        input = input.view((bs, step, num_joints, 3))
        input = input.permute(0, 3, 2, 1).contiguous()

        # velocity
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
        # position
        pos = self.joint_embed(input)
        tem1 = self.tem_embed(self.tem)
        spa1 = self.spa_embed(self.spa)
        dif = self.dif_embed(dif)
        dy = pos + dif


        # Joint-level Module
        input = torch.cat([dy, spa1], 1)
        # input = dy
        # print(input.size())
        # [64, 256, 25, 20]

        # g -- 经过softmax归一化的代表关节点直接联系的邻接矩阵
        g1 = self.compute_g1(input)
        input1 = input
        input = self.gcn1(input, g1)

        # g2 = self.compute_g2(input)
        # g2 = g2 + g1
        input = self.gcn2(input, g1)

        # g3 = self.compute_g3(input)
        # g3 = g3 + g2
        input = self.gcn3(input, g1)

        # input = input + self.bn_residual(self.residual(input1))
        # 64, 256, 25, 20

        # Frame-level Module
        input = input + tem1

        # input = self.tem_attention(input)
        input = self.cnn(input)

        # Classification
        output = self.maxpool(input)

        # 将多维数据展开，1--保持第一维不变
        output = torch.flatten(output, 1)

        output = self.fc(output)

        return output

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot

    def gen_bone(self, input):
        # input = input.permute(0, 3, 2, 1).contiguous()
        n, c, v, t = input.size()
        input1 = input[:, :c, :, :]
        if self.case == 0:
            mode = 'ntu/cs'
        else:
            mode = 'ntu/cv'
        for v1, v2 in paris[mode]:
            v1 -= 1
            v2 -= 1
            input1[:, :, v1, :] = input[:, :, v1, :] - input[:, :, v2, :]
        return input1


class norm_data(nn.Module):
    def __init__(self, dim=64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim * 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class embed(nn.Module):
    def __init__(self, dim=3, dim1=128, norm=True, bias=False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        # self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        # self.shift_in = Shift(channel=dim1, stride=1, init_scale=1)
        self.cnn1 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

        # self.shift_out = Shift(channel=dim2, stride=1, init_scale=1)

    def forward(self, x):
        x = self.cnn1(x)
        return x


class local(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        # self.shift_in = Shift(channel=dim1, stride=1, init_scale=1)
        self.bn1_1 = nn.BatchNorm2d(dim1)
        # self.bn1_2 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        # self.shift_out = Shift(channel=dim2, stride=1, init_scale=1)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

        self.residual = cnn1x1(dim1, dim2)
        self.bn_re = nn.BatchNorm2d(dim2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x0 = x1
        # CNN-1
        # x = self.bn1_1(x1)
        x = self.cnn1(x1)
        # x = self.shift_in(x1)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # CNN-2
        # x = self.bn1_2(x)
        x = self.cnn2(x)
        # x = self.shift_out(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x + self.bn_re(self.residual(x0))


class spa_attention(nn.Module):
    def __init__(self, out_channels, num_joints=25):
        super(spa_attention, self).__init__()
        # spatial attention
        ker_jpt = num_joints - 1 if not num_joints % 2 else num_joints
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)

        self.fai = cnn1x1(out_channels, out_channels)

        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, y):
        y1 = y
        y2 = self.fai(y1)

        # spatial attention
        se = y.mean(-1)  # N C V
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-1) + y

        return self.bn(y + y2)


class tem_attention(nn.Module):
    def __init__(self, out_channels):
        super(tem_attention, self).__init__()

        # temporal attention
        self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)

        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)

        self.att = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, y):
        # y1 = self.att(y)
        # return self.bn(y1)
        y1 = y
        y2 = self.att(y1)
        # temporal attention
        se = y.mean(-2)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-2) + y
        return self.bn(y + y2)


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)
        # self.spa_att = spa_attention(out_feature)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))

        # spa attention
        # x = self.spa_att(x)
        return x


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class compute_g_spa(nn.Module):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        # self.spa_attention1 = spa_attention(self.dim1)
        # self.spa_attention2 = spa_attention(self.dim1)

        # new
        # num_joints = 25
        # ker_jpt = num_joints - 1 if not num_joints % 2 else num_joints
        # pad = (ker_jpt - 1) // 2
        # self.conv_sa1 = nn.Conv1d(dim1, 1, ker_jpt, padding=pad)
        # # self.conv_sa2 = nn.Conv1d(dim1, 1, ker_jpt, padding=pad)
        #
        # nn.init.xavier_normal_(self.conv_sa1.weight)
        # # nn.init.xavier_normal_(self.conv_sa2.weight)
        # nn.init.constant_(self.conv_sa1.bias, 0)
        # # nn.init.constant_(self.conv_sa2.bias, 0)
        # self.sigmoid1 = nn.ReLU()
        # self.sigmoid2 = nn.Sigmoid()

        # tem
        # self.conv_ta = nn.Conv1d(dim1, 1, 9, padding=4)
        # nn.init.constant_(self.conv_ta.weight, 0)
        # nn.init.constant_(self.conv_ta.bias, 0)

        # self.sigmoid = nn.Sigmoid()

        self.bn1 = nn.BatchNorm2d(dim2)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        g1 = self.bn1(self.g1(x1)).permute(0, 3, 2, 1).contiguous()
        g2 = self.bn2(self.g2(x1)).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)

        # x2 = x1

        # se1 = x1.mean(-1)  # N C V
        # se11 = self.sigmoid1(self.conv_sa1(se1))
        # # bn
        # x1 = self.bn1(x1 * se11.unsqueeze(-1))
        # x1 = x1.permute(0, 3, 1, 2).contiguous()

        # se2 = x2.mean(-1)  # N C V
        # se22 = self.sigmoid2(self.conv_sa2(se2))
        # x2 = self.bn2(x2 * se22.unsqueeze(-1))
        # x2 = x2.permute(0, 3, 1, 2).contiguous()

        # tem
        # se = x2.mean(-2)
        # se1 = self.sigmoid(self.conv_ta(se))
        # x2 = x2 * se1.unsqueeze(-2)
        # x2 = x2.permute(0, 3, 1, 2).contiguous()

        # g = g1.matmul(x1)
        # g = self.softmax(g)
        return g


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
    fit.add_fit_args(parser)
    parser.set_defaults(
        network='SGN',
        dataset='NTU',
        case=0,
        batch_size=64,
        max_epochs=120,
        monitor='val_acc',
        lr=0.001,
        weight_decay=0.0001,
        lr_factor=0.1,
        workers=16,
        print_freq=20,
        train=0,
        seg=20,
    )
    args = parser.parse_args()
    model = SGN(60, 'NTU', 20, args)
    print(model)
    print(get_n_params(model))
