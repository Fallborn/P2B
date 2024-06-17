_base_ = [
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]
model = dict(
    type='ATSS',
    # pretrained="/home/lxz/P2BNet/TOV_mmdetection_cache/work_dir/coco/pre_trained/ATSS.pth",
    # pretrained='open-mmlab://resnext101_64x4d',
    # backbone=dict(
    #     type='ResNeXt',
    #     depth=101,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
          # 这里本来就是不要的
    #     # init_cfg=dict(
    #     #     type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d'),
    #     groups=64,
    #     base_width=4,
    #     dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
    #     stage_with_dcn=(False, False, True, True)),
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=6,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(
#     type='AdamW',
#     lr=5e-05,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys=dict(
#             absolute_pos_embed=dict(decay_mult=0.0),
#             relative_position_bias_table=dict(decay_mult=0.0),
#             norm=dict(decay_mult=0.0))))

checkpoint_config=dict(interval=2,filename_tmpl='detector_epoch_{}.pth')
# dataset settings
dataset_type = 'CocoFmtDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file='/home/lxz/P2BNet/TOV_mmdetection_cache/work_dir/coco/coco_1200_latest_pseudo_ann_neu_det.json',
        # ann_file='/home/lxz/P2BNet/TOV_mmdetection/data/coco/annotations/instances_train.json',
        img_prefix=data_root + 'images' + '/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/home/lxz/P2BNet/TOV_mmdetection/data/coco/annotations/instances_test.json',
        img_prefix=data_root + 'images' + '/test',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/home/lxz/P2BNet/TOV_mmdetection/data/coco/annotations/instances_test.json',
        img_prefix=data_root + 'images' + '/test',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
