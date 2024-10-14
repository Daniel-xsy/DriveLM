## Visualize the keyframe in the DriveLM Subset
## Interative visualization

import os
import cv2
import nuscenes

CAM = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

def main(save_folder):
    nusc = nuscenes.NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=True)
    user_input = input("Please enter the key frame token (q to quit): ")
    while user_input != 'q':
        key_frame = nusc.get('sample', user_input)
        tokens = []
        for cam in CAM:
            tokens.append(key_frame['data'][cam])
        for cam, token in zip(CAM, tokens):
            data_path, boxes, camera_intrinsic = nusc.get_sample_data(token)
            image = cv2.imread(data_path)
            save_path = f'{save_folder}/{user_input}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(f'{save_path}/{cam}.png', image)
        user_input = input("Please enter the key frame token (q to quit): ")


if __name__ == '__main__':
    save_folder = './example'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    main(save_folder)