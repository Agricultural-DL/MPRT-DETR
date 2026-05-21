# Modified from https://github.com/Jongchan/attention-module
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Conv_GN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 gn=True, bias=False):
        super(Conv_GN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv_BN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = Conv_BN(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return scale

class CCM(nn.Module):
    """the same structure as bounding box regression subnet
    but the output channel is different from that in regression subnet """
    def __init__(self, num_features_in, num_classes=4, prior=0.01, feature_size=256):
        super(CCM, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.1),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size, eps=1e-6),
            nn.Flatten(),
            # 第一个全连接层：将 256 维映射到 hidden_dim 维
            nn.Linear(feature_size, 4),
        )

        #self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x[-1])
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        return out

class CGFE(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['lse'], no_spatial=False,
                 num_feature_levels=4):
        super(CGFE, self).__init__()
        self.num_feat = num_feature_levels
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.conv1 = Conv_BN(256, 256, kernel_size=3, stride=2, padding=1)
        #self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Linear(256, 1)
        # self.fc = nn.Sequential(
        #     # 全局平均池化，将输入 [256, 20, 20] 转换为 [256, 1, 1]
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     # 展平操作，将 [256, 1, 1] 转换为 256 维向量
        #     nn.Flatten(),
        #     # 第一个全连接层：将 256 维映射到 hidden_dim 维
        #     nn.Linear(256, 4),
        #     # # 批归一化
        #     # nn.BatchNorm1d(100),
        #     # # ReLU 激活
        #     # nn.ReLU(inplace=True),
        #     # # 第二个全连接层：将 hidden_dim 映射到 num_classes 维
        #     # nn.Linear(100, 4)
        # )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.1),
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512, eps=1e-6),
            nn.Flatten(),
            # 第一个全连接层：将 256 维映射到 hidden_dim 维
            nn.Linear(512, 4),
        )

    def forward(self, x,):
        feats = []
        for i in range(self.num_feat):
            c2 = self.SpatialGate(x[i])
            feat1 = x[i] * c2
            c1 = self.ChannelGate(feat1)
            feat2 = feat1 * c1
            # feat = feat.flatten(2).transpose(1, 2)
            #feat = torch.cat(feat1,feat2)
            #feat = feat1 + feat2
            feats.append(feat2.to(x[i].device))
            # idx += h * w

        #x_out = torch.cat(feats, 1)
        #featsa = self.pool(x[-1])  # [batch_size, feature_dim]
        #featsa = torch.flatten(featsa, 1)  # 变成 [1, 256]
        out = self.conv1(x[-1])
        out = self.fc(out)  # [batch_size, 1]
        return feats,out


# class CGFE(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, num_feature_levels=4):
#         super(CGFE, self).__init__()
#         self.num_feat = num_feature_levels
#         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
#         self.no_spatial=no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#
#     def forward(self, x, memory, spatial_shapes):
#         feats = []
#         idx = 0
#         encoder_feat = memory.transpose(1, 2)
#         bs, c, hw = encoder_feat.shape
#
#         for i in range(self.num_feat):
#             h, w = spatial_shapes[i]
#             feat = encoder_feat[:,:,idx:idx+h*w].view(bs, 256, h, w)
#             c2 = self.SpatialGate(x[i])
#             feat = feat * c2
#             c1 = self.ChannelGate(feat)
#             feat = feat * c1
#             feat = feat.flatten(2).transpose(1, 2)
#             feats.append(feat)
#             idx += h*w
#
#         x_out = torch.cat(feats, 1)
#         return x_out


class MultiScaleFeature(nn.Module):
    def __init__(self, channels=256, is_5_scale=False):
        super(MultiScaleFeature, self).__init__()
        self.conv1 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        if is_5_scale:
            self.conv4 = Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1)
        self.is_5_scale = is_5_scale

    def forward(self, x):
        x_out = []
        x_out.append(x)
        x = self.conv1(x)
        x_out.append(x)
        x = self.conv2(x)
        x_out.append(x)
        x = self.conv3(x)
        x_out.append(x)

        if self.is_5_scale:
            x = self.conv4(x)
            x_out.append(x)
        return x_out


class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)

# class SiFE(nn.Module):
#     def __init__(self, no_spatial=False, num_feature_levels=4):
#         super(SiFE, self).__init__()
#         self.num_feat = num_feature_levels
#         self.Simam = Simam_module()
#         self.no_spatial = no_spatial
#
#     def forward(self, memory, spatial_shapes):
#         feats = []
#         #idx = 0
#         #encoder_feat = memory.transpose(1, 2)
#         #bs, c, hw = encoder_feat.shape
#
#         for i in range(self.num_feat):
#             #h, w = spatial_shapes[i]
#             #feat = encoder_feat[:, :, idx:idx + h * w].view(bs, 256, h, w)
#             feat = self.Simam(feat[i])
#             #feat = feat.flatten(2).transpose(1, 2)
#             feats.append(feat)
#             #idx += h * w
#
#         x_out = torch.cat(feats, 1)
#         return x_out