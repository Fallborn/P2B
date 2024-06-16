
python huicv/coarse_utils/noise_data_utils.py "generate_noisept_dataset"     "data/coco/annotations/instances_train.json"  "./qc_instances_2017_coarse_with_gt_train.json"  --rand_type 'range_gaussian'  --range_gaussian_sigma "0"  --size_range "0"

python huicv/coarse_utils/noise_data_utils.py "generate_noisept_dataset"     "data/coco/annotations/instances_test.json"  "./qc_instances_2017_coarse_with_gt_test.json"  --rand_type 'range_gaussian'  --range_gaussian_sigma "0"  --size_range "0"

python huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" "./qc_instances_2017_coarse_with_gt_train.json" "./qc_instances_2017_coarse_with_gt_train.json" --pseudo_w 64  --pseudo_h 64

python huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" "./qc_instances_2017_coarse_with_gt_test.json" "./qc_instances_2017_coarse_with_gt_test.json" --pseudo_w 64  --pseudo_h 6

cp ./qc_instances_2017_coarse_with_gt_train.json ./data/coco/annotations_qc_pt/instances_train2017_coarse.json
cp ./qc_instances_2017_coarse_with_gt_test.json ./data/coco/annotations_qc_pt/instances_test2017_coarse.json


