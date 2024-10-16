json_file = 'data/QA_dataset_nus/v1_1_val_nus_q_only.json'
data_root = './data/val_data/'
corruption_root = './data/val_data_corruption'
seed = 0

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)
# delete easy and hard, change mid key to seberity
corruptions = [
                 dict(type='Rain', severity=3),
                 dict(type='Saturate', severity=3)
            #    dict(type='CameraCrash', severity=4),
            #    dict(type='FrameLost', severity=4),
            #    dict(type='MotionBlur', severity=4),
            #    dict(type='ColorQuant', severity=3),
            #    dict(type='Brightness', severity=4),
            #    dict(type='LowLight', severity=3),
            #    dict(type='Fog', severity=4),
            #    dict(type='Snow', severity=3)
               ]