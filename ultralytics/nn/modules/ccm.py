import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


# class CategoricalCounting(nn.Module):
#     def __init__(self, cls_num=4):
#         super(CategoricalCounting, self).__init__()
#         self.ccm_cfg = [256, 256, 256,512,512]
#         self.in_channels = 256
#         self.conv1 = nn.Conv2d(256, self.in_channels, kernel_size=1)
#         self.ccm = make_layers(self.ccm_cfg, in_channels=self.in_channels, d_rate=2)
#         self.output = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.linear = nn.Linear(512, cls_num)
#
#     def forward(self, features, spatial_shapes=None):
#         features = features.transpose(1, 2)
#         bs, c, hw = features.shape
#         h, w = spatial_shapes[0][0], spatial_shapes[0][1]
#
#         v_feat = features[:, :, 0:h * w].view(bs, 256, h, w)
#         x = self.conv1(v_feat)
#         x = self.ccm(x)
#         out = self.output(x)
#         out = out.squeeze(3)
#         out = out.squeeze(2)
#         out = self.linear(out)
#
#         return out
#
#
# def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
#     layers = []
#     for v in cfg:
#         conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
#         if batch_norm:
#             layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#         else:
#             layers += [conv2d, nn.ReLU(inplace=True)]
#         in_channels = v
#     return nn.Sequential(*layers)


class TCategoricalCounting(nn.Module):
    def __init__(self, cls_num=4):
        super(TCategoricalCounting, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        # self.fc1 = nn.Linear(256, 1)
        self.fc = nn.Sequential(
            # nn.Flatten(),
            # nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(512),
            nn.Linear(512, cls_num),

        )  # Output a single number (count)

    def forward(self, features, spatial_shapes=None):
        bs, c, hw = features.shape
        h, w = spatial_shapes[0][0], spatial_shapes[0][1]
        featsa = self.pool(features.permute(0, 2, 1)).squeeze(-1)  # [batch_size, feature_dim]
        out = self.fc(featsa)  # [batch_size, 1]
        #print('out',out.shape)
        features = features.transpose(1, 2)
        x = features[:,:,0:h*w].view(bs, 256, h, w)
        return out, x

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class CategoricalCounting(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, feedforward_dim=2048, num_layers=6, cls_num=4):
        super(TCategoricalCounting, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, cls_num)

    def forward(self, features, spatial_shapes=None):
        bs, hw, c = features.shape  # [1, 8400, 256]
        h, w = spatial_shapes[0][0], spatial_shapes[0][1]
        # Add class token
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)  # [1, 8401, 256]

        # Transformer encoder
        features = self.transformer_encoder(features)

        # Extract the class token output
        cls_token_output = features[:, 0]  # [1, 256]

        # Classification head
        out = self.fc(cls_token_output)  # [1, cls_num]

        # Reorganize features
        # features = features.transpose(1, 2)
        # x = features[:, :, 0:h * w].view(bs, 256, h, w)

        return out