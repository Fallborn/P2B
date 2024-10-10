checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '../TOV_mmdetection_cache/work_dir/coco//epoch_5.pth'
resume_from = None
workflow = [('train', 1)]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
debug = False
num_stages = 2
model = dict(
    type='P2BNet',
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
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    roi_head=dict(
        type='P2BHead',
        num_stages=2,
        top_k=7,
        with_atten=False,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCInstanceMILHead',
            num_stages=2,
            with_loss_pseudo=False,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=6,
            num_ref_fcs=0,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_type='MIL',
            loss_mil1=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='binary_cross_entropy'),
            loss_mil2=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='gfocal_loss'))),
    train_cfg=dict(
        base_proposal=dict(
            base_scales=[4, 8, 16, 32, 64, 128],
            base_ratios=[
                0.3333333333333333, 0.5, 0.6666666666666666, 1.0, 1.5, 2.0, 3.0
            ],
            shake_ratio=None,
            cut_mode='symmetry',
            gen_num_neg=0),
        fine_proposal=dict(
            gen_proposal_mode='fix_gen',
            cut_mode=None,
            shake_ratio=[0.1],
            base_ratios=[1, 1.2, 1.3, 0.8, 0.7],
            iou_thr=0.3,
            gen_num_neg=500),
        rcnn=None),
    test_cfg=dict(rpn=None, rcnn=None))
dataset_type = 'CocoFmtDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(2000, 480), (2000, 576), (2000, 688), (2000, 864),
                   (2000, 1000), (2000, 1200)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
            'gt_true_bboxes'
        ])
]
test_scale = 1200
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2000, 1200),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                    'gt_anns_id', 'gt_true_bboxes'
                ])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    shuffle=None,
    train=dict(
        type='CocoFmtDataset',
        ann_file='data/coco/annotations_qc_pt/instances_train2017_coarse.json',
        img_prefix='data/coco/images/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(2000, 480), (2000, 576), (2000, 688), (2000, 864),
                           (2000, 1000), (2000, 1200)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                    'gt_true_bboxes'
                ])
        ]),
    val=dict(
        samples_per_gpu=2,
        type='CocoFmtDataset',
        ann_file='data/coco/annotations_qc_pt/instances_train2017_coarse.json',
        img_prefix='data/coco/images/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2000, 1200),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        keys=[
                            'img', 'gt_bboxes', 'gt_labels',
                            'gt_bboxes_ignore', 'gt_anns_id', 'gt_true_bboxes'
                        ])
                ])
        ],
        test_mode=False),
    test=dict(
        type='CocoFmtDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/images/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2000, 1200),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        keys=[
                            'img', 'gt_bboxes', 'gt_labels',
                            'gt_bboxes_ignore', 'gt_anns_id', 'gt_true_bboxes'
                        ])
                ])
        ]))
check = dict(stop_while_nan=False)
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=0)
work_dir = '../'
evaluation = dict(
    interval=1,
    metric='bbox',
    save_result_file=
    '../TOV_mmdetection_cache/work_dir/coco/_1200_latest_result.json',
    do_first_eval=True,
    do_final_eval=True)
gpu_ids = [3]
