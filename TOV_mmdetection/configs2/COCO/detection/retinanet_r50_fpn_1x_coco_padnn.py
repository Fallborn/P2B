_base_ = [
    '../_base_/models/retinanet_r50_fpn_padnn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
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
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/home/lxz/P2BNet/TOV_mmdetection_cache/work_dir/coco/coco_1200_latest_pseudo_ann_neu_det_origin.json',
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

evaluation = dict(interval=1, metric='bbox',do_final_eval=True)