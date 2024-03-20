import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from config import Config

cfg = Config()


class Res18Blk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(Res18Blk, self).__init__()

        # we add stride support for resbok, which is distinct from tutorials.
        self.conv1 = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(ch_out),
            # nn.ReLU(inplace=True),
            # nn.PReLU(),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(ch_out),
        )

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(ch_out)
            )
        # self.activation = nn.ReLU(inplace=True)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = self.activation(out)

        return out


class Res18Unet3D(nn.Module):

    def __init__(self, cfg=cfg):

        super(Res18Unet3D, self).__init__()

        self.down1 = Res18Blk(1, 16)
        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            Res18Blk(16, 32)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            Res18Blk(32, 64)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            Res18Blk(64, 128)
        )
        self.botton = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            Res18Blk(128, 256),
            # 上采样插值法可选参数最近邻(nearest),线性插值(linear),双线性插值(bilinear),
            # 三次线性插值(trilinear),默认是最近邻(nearest)
            nn.Upsample(scale_factor=2, mode='trilinear')
        )
        self.up1 = nn.Sequential(
            Res18Blk(384, 128),
            nn.Upsample(scale_factor=2, mode='trilinear'),
        )
        self.up2 = nn.Sequential(
            Res18Blk(192, 64),
            nn.Upsample(scale_factor=2, mode='trilinear'),
        )
        self.up3 = nn.Sequential(
            Res18Blk(96, 32),
            nn.Upsample(scale_factor=2, mode='trilinear'),
        )
        self.up4 = nn.Sequential(
            Res18Blk(48, 16),
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):

        conv1 = self.down1(x)  # ()
        # print('conv1.shape:', conv1.shape)
        conv2 = self.down2(conv1)
        # print('conv2.shape:', conv2.shape)
        conv3 = self.down3(conv2)
        # print('conv3.shape:', conv3.shape)
        conv4 = self.down4(conv3)
        # print('conv4.shape:', conv4.shape)
        conv5 = self.botton(conv4)
        # print('conv5.shape:', conv5.shape)
        conv6 = self.up1(torch.cat((conv5, conv4), dim=1))
        # print('conv6.shape:', conv6.shape)
        conv7 = self.up2(torch.cat((conv6, conv3), dim=1))
        # print('conv7.shape:', conv7.shape)
        conv8 = self.up3(torch.cat((conv7, conv2), dim=1))
        # print('conv8.shape:', conv8.shape)
        out = self.up4(torch.cat((conv8, conv1), dim=1))
        # print('out.shape:', out.shape)

        return out


# 自定义网络层——上采样层
class UpSampling3D(nn.Module):

    def __init__(self) -> None:

        super(UpSampling3D, self).__init__()

    # input_tensor: (batch_size, channels, depth, height, width)
    # scale_factor: Tuple of scaling factors for each dimension (depth, height, width)
    def upsampling_3d(input_tensor, scale_factor):

        output_tensor = F.interpolate(
            input_tensor, scale_factor=scale_factor, mode='trilinear', align_corners=False)

        return output_tensor


# 损失函数，处理输入的每个批次的每个样本不均衡问题，优先使用这个
class BatchBalancedCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(BatchBalancedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):

        # 剪枝，防止概率值过接近 0 或 1
        logits = torch.clamp(logits, self.epsilon, 1 - self.epsilon)
        # 逆sigmoid函数
        logits = torch.log(logits / (1. - logits))

        # 计算一批次数据中每个样本的正负样本比例
        beta = []
        for i in range(logits.size(0)):

            count_neg = (1. - targets[i]).sum()
            count_pos = targets[i].sum()
            beta.append(count_neg / (count_pos + count_neg))
        beta = torch.tensor(beta)

        # 计算权重
        pos_weight = beta / (1 - beta)

        # 计算一批次数据中每个样本的损失
        loss = None

        for i in range(logits.size(0)):

            if loss == None:
                loss = F.binary_cross_entropy_with_logits(
                    logits[i], targets[i],
                    pos_weight=pos_weight[i],
                    reduction='none'
                ).unsqueeze(dim=0)
            else:
                loss = torch.cat(
                    (loss, F.binary_cross_entropy_with_logits(
                        logits[i], targets[i],
                        pos_weight=pos_weight[i],
                        reduction='none'
                    ).unsqueeze(dim=0)), dim=0)

        # 对每批次中的每个样本根据其正负样本比例加权平均
        cost = 0
        for i in range(beta.size(0)):
            cost += (loss[i]*(1 - beta[i])).mean()
        cost /= logits.size(0)

        return torch.where(torch.eq(cost, 0.), 0., cost)


# 损失函数，处理每个输入的每个批次的样本不均衡问题
class BalancedCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # 将 logits 转换为概率
        # probs = torch.sigmoid(logits)

        # 剪枝，防止概率值过接近 0 或 1
        logits = torch.clamp(logits, self.epsilon, 1 - self.epsilon)
        # 逆sigmoid函数
        logits = torch.log(logits / (1. - logits))

        # 计算正负样本比例
        count_neg = (1. - targets).sum()
        count_pos = targets.sum()
        beta = count_neg / (count_pos + count_neg)

        # 计算权重
        pos_weight = beta / (1 - beta)

        # 计算带权重的二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight,
            reduction='none'
        )
        print(loss.shape)

        # 对损失进行加权平均，以平衡正负样本的损失贡献。(广播)
        loss = (loss*(1 - beta)).mean()

        # 在没有正样本的情况下，返回 0，以避免除零错误。否则，返回计算得到的损失。
        return torch.where(torch.eq(loss, 0.), 0., loss)
