"""
Loss functions used in patch generation
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import xywh2xyxy
from utils.metrics import box_iou


class Detection_CrossEntropy(nn.Module):

    def __init__(self, config):
        super(Detection_CrossEntropy, self).__init__()
        self.config = config


    def forward(self, output, label_batch, prob_threshold=0.):

        loss = torch.zeros(len(label_batch)).to(self.config.device)
        m_h, m_w = self.config.model_in_sz
        pred_boxes_all = xywh2xyxy(output[:, :, :4])
        pred_logits_all = output[:, :, 5:] * output[:, :, 4].unsqueeze(-1)
        # pred_logits_all = output[:, :, 5:]
        
        for i, label in enumerate(label_batch):
            indices = (output[i][:, 4] >= prob_threshold).nonzero().squeeze()
            pred_logits = pred_logits_all[i][indices, :]
            pred_boxes = pred_boxes_all[i][indices, :]
            class_ids = label[:, 0].long().unsqueeze(1)
            
            boxes = xywh2xyxy(label[:, 1:]).clamp(0, 1)
            boxes[:, [0, 2]] *= m_w
            boxes[:, [1, 3]] *= m_h

            iou_mask = (box_iou(boxes, pred_boxes) >= 0.5).nonzero()
            class_mask = iou_mask[:, 0]
            logit_mask = iou_mask[:, 1]
            pred_logits_masked = pred_logits[logit_mask, :]
            class_ids_masked = class_ids[class_mask].squeeze()

            losses = F.cross_entropy(pred_logits_masked, class_ids_masked)
            loss[i] += losses.sum()

        return loss.unsqueeze(0)

  

class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, config):
        super(MaxProbExtractor, self).__init__()
        self.config = config

    def forward(self, output: torch.Tensor):
        """
        output must be of the shape [batch, -1, 5 + num_cls]
        """
        # get values neccesary for transformation
        assert (output.size(-1) == (5 + self.config.n_classes))

        # labels = nn.functional.one_hot(torch.tensor(16), num_classes=80)

        class_confs = output[:, :, 5:5 + self.config.n_classes]  # [batch, -1, n_classes]
        objectness_score = output[:, :, 4]  # [batch, -1, 5 + num_cls] -> [batch, -1], no need to run sigmoid here

        if self.config.objective_class_id is not None:
            # norm probs for object classes to [0, 1]

            # class_confs = torch.nn.Softmax(dim=2)(class_confs)
            # only select the conf score for the objective class
            class_confs = class_confs[:, :, self.config.objective_class_id]
        else:
            # get class with highest conf for each box if objective_class_id is None
            class_confs = torch.max(class_confs, dim=2)[0]  # [batch, -1, 4] -> [batch, -1]

        confs_if_object = self.config.loss_target(objectness_score, class_confs)
        max_conf, _ = torch.max(confs_if_object, dim=1)
        # loss = nn.functional.cross_entropy(class_confs, labels)

        return max_conf

class Detection_Loss(nn.Module):

    def __init__(self, config):
        super(Detection_Loss, self).__init__()
        self.config = config
    

    def nms(self, boxes, scores, threshold, eps=1e-7):
        """
        NMS (Non-Maximum Suppression) 算法

        参数：
            boxes: 边界框坐标，形状为 (N, 4)，N为边界框的数量
            scores: 每个边界框的置信度得分，形状为 (N,)
            threshold: IoU 阈值，用于筛选重叠度高的边界框

        返回：
            keep: 保留的边界框的索引列表
        """
        # 获取边界框的坐标
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # 计算每个边界框的面积
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 按照置信度降序排列
        _, indices = scores.sort(descending=True)
        keep = []

        while indices.numel() > 0:
            # 保留置信度最高的边界框
            i = indices[0]
            keep.append(i.item())

            # 计算置信度最高的边界框与其他边界框的IoU
            xx1 = torch.max(x1[i], x1[indices[1:]])
            yy1 = torch.max(y1[i], y1[indices[1:]])
            xx2 = torch.min(x2[i], x2[indices[1:]])
            yy2 = torch.min(y2[i], y2[indices[1:]])

            w = torch.clamp(xx2 - xx1 + 1, min=0)
            h = torch.clamp(yy2 - yy1 + 1, min=0)
            inter = w * h

            iou = inter / (area[i] + area[indices[1:]] - inter + eps)

            # 删除IoU大于阈值的边界框
            indices = indices[1:][iou <= threshold]

        return keep


    def forward(self, output, label_batch, prob_threshold=0.):

        loss = torch.zeros(len(label_batch)).to(self.config.device)
        m_h, m_w = self.config.model_in_sz
        pred_boxes_all = xywh2xyxy(output[:, :, :4])      # batch_size x 25200 x 4
        pred_logits_all = output[:, :, 5:]     # batch_size x 25200 x 80
        # pred_logits_all = output[:, :, 5:] * output[:, :, 4].unsqueeze(-1)     # batch_size x 25200 x 80
        # pred_logits_all = output[:, :, 5:]
        # pred_logits_all = nn.Softmax(dim=2)(pred_logits_all)
        pred_logits_all = pred_logits_all * output[:, :, 4].unsqueeze(-1)
        pred_logits_all = pred_logits_all[:, :, self.config.objective_class_id] # batch x 25200 x 1
 
        for i, label in enumerate(label_batch):
            indices = (output[i][:, 4] >= prob_threshold).nonzero().squeeze()
            pred_logits = pred_logits_all[i][indices]
            pred_boxes = pred_boxes_all[i][indices, :]
            class_ids = label[:, 0].long().unsqueeze(1)
            
            boxes = xywh2xyxy(label[:, 1:]).clamp(0, 1)
            boxes[:, [0, 2]] *= m_w
            boxes[:, [1, 3]] *= m_h

            iou_mask = (box_iou(boxes, pred_boxes) >= 0.5).nonzero()
            class_mask = iou_mask[:, 0]
            logit_mask = iou_mask[:, 1]
            pred_logits_masked = pred_logits[logit_mask]
            pred_box_masked = pred_boxes[logit_mask, :]
            keep_index = self.nms(pred_box_masked, pred_logits_masked, 0.5)
            keep_logits = pred_logits_masked[keep_index]
            loss[i] = torch.mean(keep_logits)

        return loss.unsqueeze(0)


class TotalVariationLoss(nn.Module):
    """TotalVariationLoss: calculates the total variation of a patch.
    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.
    Reference: https://en.wikipedia.org/wiki/Total_variation
    """

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Tensor of shape [C, H, W] 
        """
        # calc diff in patch rows
        tvcomp_r = torch.sum(
            torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001), dim=0)
        tvcomp_r = torch.sum(torch.sum(tvcomp_r, dim=0), dim=0)
        # calc diff in patch columns
        tvcomp_c = torch.sum(
            torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001), dim=0)
        tvcomp_c = torch.sum(torch.sum(tvcomp_c, dim=0), dim=0)
        tv = tvcomp_r + tvcomp_c
        return tv / torch.numel(adv_patch)