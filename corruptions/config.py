json_file = './data/QA_dataset_nus/drivelm_train_300_final_v2_norm.json'
data_root = './data/nuscenes/samples/'
corruption_root = './data/train_data_corruption'
seed = 0

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)
# delete easy and hard, change mid key to seberity
corruptions = [
                  dict(type='LensObstacleCorruption', severity=5),
                  dict(type='WaterSplashCorruption', severity=4),
                  dict(type='ZoomBlur', severity=4),
                  dict(type='Rain', severity=5),
                  dict(type='Saturate', severity=5),
                  dict(type='CameraCrash', severity=4),
                  dict(type='FrameLost', severity=4),
                  dict(type='MotionBlur', severity=4),
                  dict(type='ColorQuant', severity=4),
                  dict(type='Brightness', severity=4),
                  dict(type='LowLight', severity=4),
                  dict(type='Fog', severity=5),
                  dict(type='Snow', severity=4)
               ]
