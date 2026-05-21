# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torchvision
from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = 'RTDETRValidator',  # tuple or list


class RTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class."""
        super().__init__(*args, data=data, **kwargs)

    # NOTE: add stretch version load_image for RTDETR mosaic
    def load_image(self, i, rect_mode=False):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """Temporary, only for evaluation."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True)])
            transforms = Compose([])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms


class RTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Example:
        ```python
        from ultralytics.models.rtdetr import RTDETRValidator

        args = dict(model='rtdetr-l.pt', data='coco8.yaml')
        validator = RTDETRValidator(args=args)
        validator()
        ```

    Note:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def build_dataset(self, img_path, mode='val', batch=None):
        """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # Do not need threshold for evaluation as only got 300 boxes here
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            # 1. 对score进行排序，获取排序索引
            sorted_indices = score.argsort(descending=True)  # 获取按置信度降序排列的索引

            # 2. 根据排序索引重新排序pred和score，确保顺序一致
            sorted_pred = pred[sorted_indices]  # 排序后的预测框
            sorted_score = score[sorted_indices]  # 排序后的置信度分数

            # 3. 筛选出置信度大于阈值的框
            pred = sorted_pred[sorted_score > self.args.conf]  # 根据置信度筛选预测框
            outputs[i] = pred  # [idx]

        return outputs
        # bs, _, nd = preds[0].shape
        # bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        # bboxes *= self.args.imgsz
        # outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        #
        # for i, bbox in enumerate(bboxes):  # (num_boxes, 4)
        #     # 转换 bbox 格式从 [cx, cy, w, h] 到 [x1, y1, x2, y2]
        #     bbox = ops.xywh2xyxy(bbox)  # (num_boxes, 4)
        #
        #     # 获取每个框的最高类别分数和对应的类别
        #     score, cls = scores[i].max(-1)  # (num_boxes, )
        #
        #     # 置信度筛选
        #     conf_mask = score > self.args.conf
        #     bbox, score, cls = bbox[conf_mask], score[conf_mask], cls[conf_mask]
        #
        #     # 按置信度排序
        #     sorted_indices = score.argsort(descending=True)
        #     bbox, score, cls = bbox[sorted_indices], score[sorted_indices], cls[sorted_indices]
        #
        #     # 应用 NMS
        #     if bbox.shape[0] > 0:
        #         keep = torchvision.ops.nms(bbox, score, self.args.iou)  # NMS 筛选
        #         bbox, score, cls = bbox[keep], score[keep], cls[keep]
        #
        #         # 拼接输出格式 [x1, y1, x2, y2, confidence, class]
        #         pred = torch.cat([bbox, score.unsqueeze(1), cls.unsqueeze(1).float()], dim=1)
        #         outputs[i] = pred
        #     else:
        #         outputs[i] = torch.zeros((0, 6), device=bbox.device)  # 空输出

        #return outputs
    # def expand_and_clamp_bboxes(self, bboxes, expand_ratio=0.1):
    #     """
    #     对 YOLO 格式的检测框 (x_center, y_center, w, h) 进行随机膨胀，并确保边界不越界 [0, 1]。
    #     只有随机选择的 10% 的框会膨胀，其余框保持不变。
    #
    #     参数:
    #         bboxes (torch.Tensor or np.ndarray): 形状为 (N, 4)，每行为 [x_center, y_center, w, h]。
    #         expand_ratio (float): 膨胀比例，例如 0.1 表示宽度和高度增加 10%。
    #
    #     返回:
    #         torch.Tensor: 形状为 (N, 4)，膨胀并修正后的检测框。
    #     """
    #     if isinstance(bboxes, np.ndarray):
    #         bboxes = torch.tensor(bboxes, dtype=torch.float32)
    #
    #     # 随机选择 10% 的框
    #     num_boxes = bboxes.size(0)
    #     random_indices = torch.rand(num_boxes) < 0.2  # 随机生成掩码，选中约 10% 的框
    #
    #     # 提取选中的框和未选中的框
    #     selected_bboxes = bboxes[random_indices]
    #     unselected_bboxes = bboxes[~random_indices]
    #
    #     # 膨胀选中的框
    #     if selected_bboxes.size(0) > 0:
    #         x_center, y_center, w, h = (
    #             selected_bboxes[:, 0],
    #             selected_bboxes[:, 1],
    #             selected_bboxes[:, 2],
    #             selected_bboxes[:, 3],
    #         )
    #         # 膨胀宽度和高度
    #         w_new = w * (1 + expand_ratio)
    #         h_new = h * (1 + expand_ratio)
    #
    #         # 计算左上角和右下角坐标
    #         x_min = x_center - w_new / 2
    #         y_min = y_center - h_new / 2
    #         x_max = x_center + w_new / 2
    #         y_max = y_center + h_new / 2
    #
    #         # 修正坐标范围到 [0, 1]
    #         x_min = torch.clamp(x_min, 0.0, 1.0)
    #         y_min = torch.clamp(y_min, 0.0, 1.0)
    #         x_max = torch.clamp(x_max, 0.0, 1.0)
    #         y_max = torch.clamp(y_max, 0.0, 1.0)
    #
    #         # 重新计算中心点、宽度和高度
    #         x_center_new = (x_min + x_max) / 2
    #         y_center_new = (y_min + y_max) / 2
    #         w_final = x_max - x_min
    #         h_final = y_max - y_min
    #
    #         # 拼接结果
    #         expanded_selected_bboxes = torch.stack([x_center_new, y_center_new, w_final, h_final], dim=1)
    #     else:
    #         expanded_selected_bboxes = selected_bboxes
    #
    #     # 合并膨胀后的框和未选中的框
    #     expanded_bboxes = torch.cat([expanded_selected_bboxes, unselected_bboxes], dim=0)
    #
    #     # 恢复原始顺序
    #     return expanded_bboxes[
    #         torch.argsort(torch.cat([random_indices.nonzero().squeeze(), (~random_indices).nonzero().squeeze()]))]
    def _prepare_batch(self, si, batch):
        idx = batch['batch_idx'] == si
        cls = batch['cls'][idx].squeeze(-1)
        bbox = batch['bboxes'][idx]
        #bbox = self.expand_and_clamp_bboxes(bbox,0.1)
        #batch['bboxes'] = batch_boxes.to(img.device)
        ori_shape = batch['ori_shape'][si]
        imgsz = batch['img'].shape[2:]
        ratio_pad = batch['ratio_pad'][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox)  # target boxes
            bbox[..., [0, 2]] *= ori_shape[1]  # native-space pred
            bbox[..., [1, 3]] *= ori_shape[0]  # native-space pred
        prepared_batch = dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)
        return prepared_batch

    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        predn[..., [0, 2]] *= pbatch['ori_shape'][1] / self.args.imgsz  # native-space pred
        predn[..., [1, 3]] *= pbatch['ori_shape'][0] / self.args.imgsz  # native-space pred
        return predn.float()
