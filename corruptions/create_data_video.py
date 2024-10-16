import argparse
import os
import cv2
import numpy as np
import torch
import mmcv
from nuscenes import NuScenes
from mmcv import Config, DictAction
from corruptions import CORRUPTIONS
import subprocess


CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate DriveLM-C Dataset')
    parser.add_argument('--config', default='corruptions/config_video.py', help='test config file path')
    args = parser.parse_args()

    return args


def save_path(corruption_root, corruption, filepath):
    """Return save path of generated corruptted images
    """
    return os.path.join(corruption_root, corruption, filepath)


def save_multi_view_img(imgs, img_filenames, root, corruption):
    """Save six view images
    Args:
        img (np.array): [M, H, W, C]
    """
    assert imgs.shape[0] == len(img_filenames), "Image size do not equal to filename size"
    imgs = np.squeeze(imgs)

    for i in range(len(imgs)):
        filepath = os.path.join(root, corruption, img_filenames[i])
        mmcv.imwrite(imgs[i], filepath)


def load_images(cfg, img_paths):
    imgs = []
    img_paths_new = []
    for cam, img_path in img_paths.items():
        img_path = img_path.replace('../nuscenes/samples/', cfg.data_root)
        img = mmcv.imread(img_path)
        imgs.append(img)
        img_paths_new.append(img_path.replace(cfg.data_root, ''))
    imgs = np.array(imgs)
    return imgs, img_paths_new


def images_to_video(frames, output_path='temp_in.mp4', fps=30):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        out.write(frame)
    out.release()


def video_to_frames(video_path='temp_out.mp4'):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    set_seed(cfg.seed)
    json_data = mmcv.load(cfg.json_file)

    print('Begin generating video corruption')
    for corruption in cfg.corruptions:
        print(f'\n### Corruption type: {corruption.type}\n')  

        corrupt = CORRUPTIONS.build(dict(type=corruption.type, severity=corruption.severity, norm_config=cfg.img_norm_cfg))

        prog_bar = mmcv.ProgressBar(len(json_data))
        for i, (scene_token, values) in enumerate(json_data.items()):
            key_frames = values['key_frames']
            img_paths_all = dict()
            imgs_all = dict()
            for cam in CAMS:
                imgs_all[cam] = []
            for key_frame_token, data_item in key_frames.items():
                img_paths = data_item['image_paths']
                # Load images from nuscenes dataset
                imgs, img_paths = load_images(cfg, img_paths)
                img_paths_all[key_frame_token] = img_paths
                for cam, img in zip(CAMS, imgs):
                    imgs_all[cam].append(img)
            # Convert to videos
            imgs_all_corrupt = dict()
            for cam in CAMS:
                imgs_all_corrupt[cam] = []
            corrupted_imgs = dict()
            for cam in CAMS:
                images_to_video(imgs_all[cam], output_path=f'temp_in_{cam}.mp4', fps=30)
                # Apply corruption
                corrupt(src=f'temp_in_{cam}.mp4', dst=f'temp_out_{cam}.mp4')
                imgs_all_corrupt[cam] = video_to_frames(video_path=f'temp_out_{cam}.mp4')
            # save images
            for i, (key_frame_token, img_paths) in enumerate(img_paths_all.items()):
                new_img = np.array([imgs_all_corrupt[cam][i] for cam in CAMS])
                save_multi_view_img(new_img, img_paths, cfg.corruption_root, corruption.type)
            prog_bar.update()


if __name__ == '__main__':
    main()