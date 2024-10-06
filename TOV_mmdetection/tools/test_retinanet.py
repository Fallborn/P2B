# mmdet常见的推理api
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
# mmcv是mmdet之类的通用库
import mmcv

config_file = 'configs2/COCO/detection/retinanet_r50_fpn_1x_coco_padnn.py'
# checkpoint 需要自己手动下载，Model Zoo里应该有
checkpoint_file = 'work_dirs/retinanet_r50_fpn_1x_coco_padnn/latest.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:3')  # or device='cuda:0'

img = 'data/coco/images/train/pitted_surface_190.jpg'


result = inference_detector(model,img)


model.show_result(img, result, out_file='./result.jpg')