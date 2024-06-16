_base_ = './atss_r50_fpn_1x_coco_neu_det.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(depth=101),
)
