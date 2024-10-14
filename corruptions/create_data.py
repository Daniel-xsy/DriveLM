import argparse
import os
import numpy as np
import torch
import mmcv
from nuscenes import NuScenes
from mmcv import Config, DictAction
from corruptions import CORRUPTIONS


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate DriveLM-C Dataset')
    parser.add_argument('--config', default='corruptions/config.py', help='test config file path')
    args = parser.parse_args()

    return args


def save_path(corruption_root, corruption, filepath):
    """Return save path of generated corruptted images
    """
    return os.path.join(corruption_root, corruption, filepath)


def save_multi_view_img(imgs, img_filenames, root, corruption):
    """Save six view images
    Args:
        img (np.array): [B, M, H, W, C]
    """
    assert imgs.shape[0] == 1, "Only support batchsize = 1"
    assert imgs.shape[1] == len(img_filenames), "Image size do not equal to filename size"
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
    imgs = torch.FloatTensor(imgs).permute(0, 3, 1, 2).unsqueeze(0)
    return imgs, img_paths_new


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    set_seed(cfg.seed)
    json_data = mmcv.load(cfg.json_file)

    print('Begin generating nuScenes-C dataset')
    for corruption in cfg.corruptions:
        print(f'Corruption type: {corruption.type}')  

        corrupt = CORRUPTIONS.build(dict(type=corruption.type, severity=corruption.severity, norm_config=cfg.img_norm_cfg))

        prog_bar = mmcv.ProgressBar(len(json_data))
        for i, (scene_token, values) in enumerate(json_data.items()):
            key_frames = values['key_frames']
            for key_frame_token, data_item in key_frames.items():
                QAs = data_item['QA']
                img_paths = data_item['image_paths']
                # Load images from nuscenes dataset
                imgs, img_paths = load_images(cfg, img_paths)
                new_img = corrupt(imgs)
                new_img = new_img.astype(np.uint8)
                
                save_multi_view_img(new_img, img_paths, cfg.corruption_root, corruption.type)
            prog_bar.update()


if __name__ == '__main__':
    main()