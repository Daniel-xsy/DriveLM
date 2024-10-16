json_file = 'data/QA_dataset_nus/v1_1_val_nus_q_only.json'
data_root = './data/val_data/'
corruption_root = './data/val_data_corruption'
seed = 0

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)
# delete easy and hard, change mid key to seberity
# corruptions = [dict(type='H256ABRCompression', severity=4)]
corruptions = [dict(type='BitError', severity=1)]