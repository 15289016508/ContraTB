"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
import torch.nn.functional as F

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,config ,contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = config.temprature
        self.contrast_mode = contrast_mode
        self.base_temperature = config.base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        #features = F.normalize(features, p=2, dim=-1).to(device)
        labels=labels.to(device)
        #print("features", features.shape)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        #print(batch_size)
        # input()
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)#相同标签就是正例为1
            # mask = torch.eq(labels, 1) & torch.eq(labels.T, 1)
            # mask = mask.float().to(device)
        else:
            mask = mask.float().to(device)

        # print('labels',labels.shape,labels)
        # print("mask1",mask.shape,mask)
        # print(labels)
        # input()
        # contrast_count = features.shape[1]
        # 一个样本只有一个视图，对比样本数则为1
        contrast_count = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # contrast_feature=features
        #print("contrast_feature",contrast_feature.shape)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # print("anchor_feature",anchor_feature.shape,anchor_feature)
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)
        dot_product = torch.matmul(anchor_feature, anchor_feature.T)

        # 计算每个矩阵行的范数
        norm_anchor = torch.norm(anchor_feature, dim=1)

        # 归一化余弦相似度
        normalized_cosine_similarity = dot_product / (norm_anchor.unsqueeze(1) * norm_anchor.unsqueeze(0))

        # 使用温度参数进行缩放
        anchor_dot_contrast = normalized_cosine_similarity / self.temperature
        # print("anchor_dot_contrast2",anchor_dot_contrast.shape,anchor_dot_contrast)
        # anchor_dot_contrast = torch.log(anchor_dot_contrast.clamp(min=1e-6))  # 添加小量以防止取对数时log(0)
        # print("anchor_dot_contrast-log", anchor_dot_contrast.shape, anchor_dot_contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print("logits_max",logits_max)

        logits =  anchor_dot_contrast - logits_max.detach()
        # print("logits:",logits.shape,logits)
        # exit()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print("anchor_count",anchor_count,"contrast_count",contrast_count)
        # print(mask.shape)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # print("mask2",mask.shape,mask)
        # print("logits_mask",logits_mask)
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print("exp_logits", exp_logits)
        # print("torch.log(exp_logits.sum(1, keepdim=True))",torch.log(exp_logits.sum(1, keepdim=True)))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print("log_prob", log_prob)
        # input()

        # compute mean of log-likelihood over positive
        # print()
        # print("(mask * log_prob).sum(1):",(mask * log_prob).sum(1))
        # print("mask.sum(1)",mask.sum(1))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1))
        # input

        #(mask * log_prob).sum(1)这部分是计算每个样本的对比损失对数概率，其中 mask 用于过滤掉负样本。
        #mask.sum(1)计算对每个样本中正样本的数量进行求和。这将用作分母，以得到平均对数概率。
        # loss
        # print("mean_log_prob_pos",mean_log_prob_pos)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = - 10 * mean_log_prob_pos
        # loss = - mean_log_prob_pos
        # print(loss)
        # print("batch_size",batch_size)
        # print("anchor_count",anchor_count)
        loss = loss.view(anchor_count, batch_size).mean()
        # mean_loss_batch = torch.mean(loss, dim=0)
        # print("mean_loss_batch",mean_loss_batch)
        #print("loss",loss.size ,loss)
        return loss
