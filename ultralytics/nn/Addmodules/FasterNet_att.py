# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os
import torchvision
from ultralytics.nn.modules import ADown

from .EMAttention import *

from .iAT import *

import cv2
import numpy as np
from ..modules.cgfe import CGFE
__all__ = ['FasterNet_A','FasterNet_A1']

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        # self.partial_conv3 = nn.Sequential(
        #         nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False),
        #         nn.BatchNorm2d(self.dim_conv3),
        #         nn.ReLU(inplace=True),
        #         #EMA(self.dim_conv3)
        #     )

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),

        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        self.ema = EMA(dim)
        #self.ema = SCSA(dim)
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.ema(self.drop_path(self.mlp(x)))
        #x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):
        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        # self.proj = nn.Sequential(
        #         nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False),
        #         nn.BatchNorm2d(embed_dim),
        #         nn.ReLU(inplace=True),
        #         EMA(embed_dim)
        #     )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        #self.reduction = ADown(dim, 2 * dim)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x

class FasterNet_A(nn.Module):

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size=4,
                 patch_stride=4,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 fork_feat=True,
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 return_idx=[2, 4, 6],
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        if not fork_feat:
            self.num_classes = num_classes
        self.return_idx = return_idx
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        #self.HAT = HAT()
        self.iAT = IAT()
        # self.iAT.load_state_dict(
        #     torch.load('/home/wjs/wjs/ultralytics-improved2/MIT_5K.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'), strict=False)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** i_stage),
                                 norm_layer=norm_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat
        #self.CGFE = CGFE(gate_channels=256, reduction_ratio=16, num_feature_levels=3, pool_types=['lp', 'lse'])
        self.forward = self.forward_det
        # add a norm layer for each output
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                raise NotImplementedError
            else:
                layer = norm_layer(int(embed_dim * 2 ** i_emb))
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()
        self.width_list = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]


    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        # enhanced_image = self.enhance_contrast_batch(x).to(x.device)
        # x = torch.cat([x, enhanced_image], dim=1)
        # x = x.to(torch.float32)
        x = self.iAT(x)

        #torchvision.utils.save_image(iat, 'output.png')

        #x = torch.cat([x, iat], dim=1)
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                #x_out = self.CGFE(x_out)
                    #outs.append(x_out)
                outs.append(x_out)
        return outs


    # def enhance_contrast_batch(self,tensor_images):
    #     """
    #     对批量 PyTorch Tensor 格式的图像进行增强（基于 OpenCV 的 CLAHE 方法）。
    #
    #     参数:
    #         tensor_images (torch.Tensor): 输入图像 Tensor，形状为 (B, C, H, W)，值范围 [0, 1]。
    #
    #     返回:
    #         torch.Tensor: 增强后的图像，形状为 (B, C, H, W)，值范围 [0, 1]。
    #     """
    #     enhanced_images = []
    #
    #     for tensor_image in tensor_images:
    #         # 单张图像的处理
    #         np_image = tensor_image.permute(1, 2, 0).cpu().numpy()  # 转为 NumPy 格式，形状 [H, W, C]
    #         np_image = (np_image * 255).astype(np.uint8)  # 转为 0-255 的 uint8 格式
    #
    #         # 转换为 Lab 色彩空间并增强
    #         lab = cv2.cvtColor(np_image, cv2.COLOR_RGB2Lab)
    #         l, a, b = cv2.split(lab)
    #         clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(16, 16))
    #         b = clahe.apply(b)
    #         l = clahe.apply(l)
    #         enhanced_lab = cv2.merge((l, b))
    #         #enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_Lab2RGB)
    #
    #         # 转换回 Tensor 格式，并恢复到 [C, H, W]
    #         tensor_enhanced = torch.from_numpy(enhanced_lab.astype(np.float32) / 255.0).permute(2, 0, 1)
    #         enhanced_images.append(tensor_enhanced)
    #
    #     # 堆叠所有增强的图像，返回形状为 (B, C, H, W)
    #     return torch.stack(enhanced_images)
def FasterNet_A1(pretrained='/home/wjs/wjs/ultralytics-improved2/FasterNett0.pth'):
    model = FasterNet_A()
    if pretrained:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(pretrained)))
    return model
def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        # k = k[9:]
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 640, 640)
    image = torch.rand(*image_size)

    # Model
    model = FasterNet_A()

    out = model(image)
    print(len(out))
