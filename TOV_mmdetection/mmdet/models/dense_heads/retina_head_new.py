import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .anchor_head import AnchorHead
import torch

class MDA(nn.Module):
    def __init__(self, channels, factor=16, ratio=4):
        super(MDA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels // self.groups, (channels // self.groups) // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d((channels // self.groups) // ratio, channels // self.groups, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        self.conv1x1_reduce = nn.Conv2d(channels // self.groups, (channels // self.groups) // 2, kernel_size=1,
                                        stride=1, padding=0)
        self.conv1x1_restore = nn.Conv2d((channels // self.groups) // 2, channels // self.groups, kernel_size=1,
                                         stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.conv3x3_dynamic = ConvModule(channels // self.groups, channels // self.groups, 3, stride=1,padding=1, conv_cfg= dict(type='DCNv2', deform_groups=1))
        self.involution = Involution(channels // self.groups, kernel_size=1, stride=1, group_channels=16, reduction_ratio=4)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w

        # First route
        x_r1 = self.involution(group_x)
        x_r1 = self.conv1x1_reduce(x_r1)  # dimension reduction, now b*g, c//g//2, h, w
        x_h = self.pool_h(x_r1)
        x_w = self.pool_w(x_r1).permute(0, 1, 3, 2)
        hw = self.conv1x1_restore(torch.cat([x_h, x_w], dim=2))  # restore dimension, now b*g, c//g, h, w
        hw = hw.sigmoid()  # apply sigmoid
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # Element-wise multiplication with the initial group_x
        out = group_x * x_h.sigmoid()  # this shape is same as group_x

        # Channel wise attention as described in CA
        avg_out = self.fc(self.avg_pool(out))
        max_out = self.fc(self.max_pool(out))
        out_ca = avg_out + max_out
        out_ca = self.sigmoid(out_ca)  # Channel wise coefficient

        # Second route
        x_r2 = self.conv3x3_1(group_x)
        x_r2 = self.gn(x_r2)
        x_r2 = self.conv3x3_dynamic(x_r2)

        # Multiply with coefficient
        x_r2 = out_ca.expand_as(x_r2) * x_r2

        # Combine routes
        out = out + x_r2

        weights = out.sigmoid()

        return (group_x * weights).reshape(b, c, h, w)  # back to the original shape

class Involution(nn.Module):
    def __init__(self, channels, kernel_size=7, stride=1, group_channels=16, reduction_ratio=2):
        super().__init__()
        # assert not (channels % group_channels or channels % reduction_ratio)

        # in_c=out_c
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        # 每组多少个通道
        self.group_channels = group_channels
        self.groups = channels // group_channels

        # reduce channels
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU(inplace=False)
        )
        # span channels
        self.span = nn.Conv2d(
            channels // reduction_ratio,
            self.groups * kernel_size ** 2,
            1
        )

        self.down_sample = nn.AvgPool2d(stride) if stride != 1 else nn.Identity()
        self.unfold = nn.Unfold(kernel_size, padding=(kernel_size - 1) // 2, stride=stride)

    def forward(self, x):
        # generate involution kernel: (b,G*K*K,h,w)
        weight_matrix = self.span(self.reduce(self.down_sample(x)))
        b, _, h, w = weight_matrix.shape

        # unfold input: (b,C*K*K,h,w)
        x_unfolded = self.unfold(x)
        # (b,C*K*K,h,w)->(b,G,C//G,K*K,h,w)
        x_unfolded = x_unfolded.view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)

        # (b,G*K*K,h,w) -> (b,G,1,K*K,h,w)
        weight_matrix = weight_matrix.view(b, self.groups, 1, self.kernel_size ** 2, h, w)
        # clear the nan value
        epsilon = torch.finfo(x_unfolded.dtype).eps
        mul_add = (weight_matrix * x_unfolded + epsilon).sum(dim=3)

        out = mul_add.view(b, self.channels, h, w)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

@HEADS.register_module()
class RetinaHeadPADNN(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        super(RetinaHeadPADNN, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.mda_cls = MDA(self.feat_channels,1,16)
        self.mda_reg = MDA(self.feat_channels,1,16)
        self.ca_cls = ChannelAttention(self.feat_channels)
        self.ca_reg = ChannelAttention(self.feat_channels)
        self.cbam_cls = CBAM(self.feat_channels)
        self.cbam_reg = CBAM(self.feat_channels)

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.cls_convs.insert(len(self.cls_convs)-1,self.mda_cls)
        self.reg_convs.insert(len(self.reg_convs)-1,self.mda_reg)
        # self.cls_convs.append(self.mda_cls)
        # self.reg_convs.append(self.mda_reg)
        # self.cls_convs.append(self.ca_cls)
        # self.reg_convs.append(self.ca_reg)
        # self.cls_convs.append(self.cbam_cls)
        # self.reg_convs.append(self.cbam_reg)

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for i, cls_conv in enumerate(self.cls_convs):
            if i != len(self.cls_convs)-1:
                cls_feat = cls_feat + cls_conv(cls_feat)

            else:
                cls_feat = cls_feat + 0.00001*cls_conv(cls_feat)

        for i, reg_conv in enumerate(self.reg_convs):
            reg_feat = reg_feat + reg_conv(reg_feat)

        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
