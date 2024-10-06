# mmdet常见的推理api
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
# mmcv是mmdet之类的通用库
import mmcv

config_file = 'configs2/COCO/detection/retinanet_r50_fpn_1x_coco_pcb.py'
# checkpoint 需要自己手动下载，Model Zoo里应该有
checkpoint_file = 'work_dirs/retinanet_r50_fpn_1x_coco_pcb/epoch_12.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:3')  # or device='cuda:0'

# neu-det
# crazing
# inclusion
# patches
# pitted_surface
# rolled-in_scale
# scratches
# classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# open
# short
# mousebite
# spur
# copper
# Pin-hole


classes = ['missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']
for cls in classes:
    for i in range(10):
        try:
            img = 'data/coco_pcb/images/train/01_{}_0{}.jpg'.format(cls,i)


            result = inference_detector(model,img)


            model.show_result(img, result, out_file='./{}_{}.jpg'.format(cls,i),score_thr=0.7)
        except:
            pass