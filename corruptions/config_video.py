json_file = './data/QA_dataset_nus/drivelm_train_300_final_v2_norm.json'
data_root = './data/nuscenes/samples/'
corruption_root = './data/train_data_corruption'
seed = 0

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)
# delete easy and hard, change mid key to seberity
corruptions = [dict(type='H256ABRCompression', severity=4),
               dict(type='BitError', severity=2]