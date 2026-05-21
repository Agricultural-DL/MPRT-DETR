import torch
import torch.nn as nn
import torch.nn.functional as F
 
def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))
 
def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model
 
 
class MimicLoss(nn.Module):
    def __init__(self, channels_s, channels_t):
        super(MimicLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mse = nn.MSELoss()
 
    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.mse(s, t))
        loss = sum(losses)
        return loss
 
 
class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """
 
    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau
 
    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
 
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
 
            N, C, H, W = s.shape
 
            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]
 
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)
 
            losses.append(cost / (C * N))
        loss = sum(losses)
 
        return loss
 
 
class MGDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
 
        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel in channels_t
        ]
 
    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss
 
    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
 
        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)
 
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)
        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss
 
 
class Distill_LogitLoss:
    def __init__(self, p, t_p, alpha=0.25):
        t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
        self.p = p
        self.t_p = t_p
        self.logit_loss = t_ft([0])
        self.DLogitLoss = nn.MSELoss(reduction="none")
        self.bs = p[0].shape[0]
        self.alpha = alpha
 
    def __call__(self):
        # per output
        assert len(self.p) == len(self.t_p)
        for i, (pi, t_pi) in enumerate(zip(self.p, self.t_p)):  # layer index, layer predictions
            assert pi.shape == t_pi.shape
            self.logit_loss += torch.mean(self.DLogitLoss(pi, t_pi))
        return self.logit_loss[0] * self.alpha
 
 
# def get_fpn_features(x, model, fpn_layers=[15, 18, 21]):
#     y, fpn_feats = [], []
#     with torch.no_grad():
#         model = de_parallel(model)
#         module_list = model.model[:-1] if hasattr(model, "model") else model[:-1]
#         for m in module_list:
#             # if not from previous layer
#             if m.f != -1:
#                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
#             x = m(x)
#             y.append(x if m.i in model.save else None)  # save output
#             if m.i in fpn_layers:
#                 fpn_feats.append(x)
#     return fpn_feats
def get_fpn_features(x, model, fpn_layers=[15, 18, 21]):
    """获取指定层的特征图"""
    y, fpn_feats = [], []

    with torch.no_grad():
        model = de_parallel(model)
        module_list = model.model[:-1] if hasattr(model, "model") else model[:-1]

        for m in module_list:
            # 处理输入来源
            #print(m.f)
            if m.f != -1:  # 如果不是从上一层获取输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            # 处理具有backbone属性的模块
            if hasattr(m, 'backbone'):
                try:
                    # 调用backbone模块
                    x = m(x)
                    # 如果backbone输出不是5个元素，插入None
                    if len(x) != 5:
                        x.insert(0, None)

                    for index, i in enumerate(x):
                        if index in [2,3,6,7,8,8,9,12,15,18,21]:
                            # if (index+1) in self.save:
                            y.append(i)
                        else:
                            y.append(None)

                    # 最后一个输出作为下一层的输入
                    x = x[-1]

                except Exception as e:
                    print(f"Error processing backbone module {m}: {e}")
                    raise

            else:
                try:
                    x = m(x)
                    y.append(x if m.i in [2,3,6,7,8,8,9,12,15,18,21] else None)  # save output
                except Exception as e:
                    print(f"Error processing module {m}: {e}")
                    raise


            # 检查是否需要保存特征图
            if hasattr(m, 'i') and m.i in fpn_layers:
                if x is not None:
                    fpn_feats.append(x)
                else:
                    print(f"Warning: Layer {m.i} output is None")

    return fpn_feats
# def get_channels(model, fpn_layers=[15, 18, 21]):
#     y, out_channels = [], []
#     p = next(model.parameters())
#     x = torch.zeros((1, 3, 64, 64), device=p.device)
#     with torch.no_grad():
#         model = de_parallel(model)
#         module_list = model.model[:-1] if hasattr(model, "model") else model[:-1]
#
#         for m in module_list:
#             # if not from previous layer
#             if m is 'FasterNet':
#
#             if m.f != -1:
#                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
#             x = m(x)
#             y.append(x if m.i in model.save else None)  # save output
#             if m.i in fpn_layers:
#                 out_channels.append(x.shape[1])
#     return out_channels
def get_channels(model, fpn_layers=[15, 18, 21]):
    y, out_channels = [], []
    p = next(model.parameters())
    x = torch.zeros((1, 3, 64, 64), device=p.device)
    with torch.no_grad():
        model = de_parallel(model)
        module_list = model.model[:-1] if hasattr(model, "model") else model[:-1]

        for m in module_list:
            # 处理输入来源
            if m.f != -1:  # 如果不是从上一层获取输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # 处理具有backbone属性的模块
            if hasattr(m, 'backbone'):
                # 调用backbone模块
                backbone_output = m(x)

                # 如果backbone输出不是5个元素，插入None
                if len(backbone_output) != 5:
                    backbone_output.insert(0, None)

                # 将backbone输出添加到y列表
                for index, i in enumerate(backbone_output):
                    if index in [1,2,3,4]:  # 注意：这里使用的是m.save，不是model.save
                        y.append(i)
                    else:
                        y.append(None)

                # 最后一个输出作为下一层的输入
                x = backbone_output[-1]

                # 如果m.i在fpn_layers中，记录输出通道
                if hasattr(m, 'i') and m.i in fpn_layers:
                    out_channels.append(x.shape[1])

            else:
                # 处理普通模块
                try:
                    if hasattr(m, 'input_nums') and m.input_nums > 1:
                        # 输入数量多于一个
                        x = m(*x)
                    else:
                        x = m(x)
                except AttributeError:
                    # 如果模块没有input_nums属性，直接调用
                    x = m(x)

                # 保存输出
                if hasattr(m, 'i') and hasattr(model, 'save'):
                    if m.i in model.save:
                        y.append(x)
                    else:
                        y.append(None)
                else:
                    y.append(None)

                # 如果m.i在fpn_layers中，记录输出通道
                if hasattr(m, 'i') and m.i in fpn_layers:
                    out_channels.append(x.shape[1])

    return out_channels
 
class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='cwd'):
        super(FeatureLoss, self).__init__()
 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1, padding=0).to(device)
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]
 
        if distiller == 'mimic':
            self.feature_loss = MimicLoss(channels_s, channels_t)
 
        elif distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)
 
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError
 
    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []
 
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)
 
        loss = self.feature_loss(stu_feats, tea_feats)
        return loss
