from copy import deepcopy
import functools
import PIL
from PIL import Image

import torch
import numpy as np

import json
import cv2
import subprocess
from mmcv.utils import Registry
from imagecorruptions import corrupt


CORRUPTIONS= Registry('corruptions')


@CORRUPTIONS.register_module()
class Clean:
    def __init__(self, severity, norm_config):
        """
        No corruption
        """
        self.severity = severity
        assert severity >= 1 and severity <= 5, f"Corruption Severity should between (1, 5), now {severity}"
        self.norm_config = norm_config
        self.corruption = 'clean'
    def __call__(self, x):
        return x


@CORRUPTIONS.register_module()
class BaseCorruption:
    def __init__(self, severity, norm_config, corruption):
        """
        Base Corruption Class
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        self.severity = severity
        assert severity >= 1 and severity <= 5, f"Corruption Severity should between (1, 5), now {severity}"
        self.norm_config = norm_config
        self.corruption = corruption
        try:
            self.corrupt_func = self._get_corrupt_func()
        except:
            self.corrupt_func = None

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): [B, M, C, H, W]
        """
        mean = self.norm_config['mean']
        std = self.norm_config['std']
        
        img = deepcopy(img)
        B, M, C, H, W = img.size()
        img = img.permute(0, 1, 3, 4, 2) # [B, M, C, H, W] => [B, M, H, W, C]
        img = img * torch.tensor(std) + torch.tensor(mean)
        # pixel value [0, 255]
        assert img.min() >= 0 and img.max() <= 255, "Image pixel out of range"
        new_img = np.zeros_like(img)
        for b in range(B):
            for m in range(M):
                new_img[b, m] = self.corrupt_func(np.uint8(img[b, m].numpy()))

        # new_img = new_img.permute(0, 1, 4, 2, 3)

        return new_img

    def _get_corrupt_func(self):
        return functools.partial(corrupt, corruption_name=self.corruption, severity=self.severity)


@CORRUPTIONS.register_module()
class DefocusBlur(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Defocus Blur'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'defocus_blur')


@CORRUPTIONS.register_module()
class GlassBlur(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Glass Blur'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'glass_blur')


@CORRUPTIONS.register_module()
class MotionBlur(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Motion Blur'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'motion_blur')


@CORRUPTIONS.register_module()
class ZoomBlur(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Zoom Blur'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'zoom_blur')


@CORRUPTIONS.register_module()
class GaussianNoise(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Gaussian Noise'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'gaussian_noise')


@CORRUPTIONS.register_module()
class ImpulseNoise(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Impulse Noise'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'impulse_noise')


@CORRUPTIONS.register_module()
class ShotNoise(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'ISO Noise'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'shot_noise')


@CORRUPTIONS.register_module()
class ISONoise(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Shot Noise'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'iso_noise')
        self.corrupt_func = self._get_corrupt_func()

    def _get_corrupt_func(self):
        return functools.partial(self.iso_noise, severity=self.severity)

    def iso_noise(self, x, severity):
        c_poisson = 25
        x = np.array(x) / 255.
        x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255.
        c_gauss = 0.7 * [.08, .12, 0.18, 0.26, 0.38][severity]
        x = np.array(x) / 255.
        x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255.
        return np.uint8(x)


@CORRUPTIONS.register_module()
class Brightness(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Brightness'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'brightness')


@CORRUPTIONS.register_module()
class LowLight(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Dark'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'dark')
        self.corrupt_func = self._get_corrupt_func()

    def _get_corrupt_func(self):
        return functools.partial(self.low_light, severity=self.severity)

    def imadjust(self, x, a, b, c, d, gamma=1):
        y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
        return y

    def poisson_gaussian_noise(self, x, severity):
        c_poisson = 10 * [60, 25, 12, 5, 3][severity]
        x = np.array(x) / 255.
        x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
        c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity]
        x = np.array(x) / 255.
        x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255
        return np.uint8(x)

    def low_light(self, x, severity):
        c = [0.60, 0.50, 0.40, 0.30, 0.20][severity]
        # c = [0.50, 0.40, 0.30, 0.20, 0.10][severity-1]
        x = np.array(x) / 255.
        x_scaled = self.imadjust(x, x.min(), x.max(), 0, c, gamma=2.) * 255
        x_scaled = self.poisson_gaussian_noise(x_scaled, severity=severity)
        return x_scaled


@CORRUPTIONS.register_module()
class Fog(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Fog'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'fog')


@CORRUPTIONS.register_module()
class Snow(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Snow'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'snow')


@CORRUPTIONS.register_module()
class CameraCrash:
    def __init__(self, severity, norm_config):
        """ 
        Create corruptions: 'Camera Crash'.
        """
        self.severity = severity
        assert severity >= 1 and severity <= 5, f"Corruption Severity should between (1, 5), now {severity}"
        self.norm_config = norm_config
        self.corruption = 'cam_crash'
        self.crash_camera = self.get_crash_camera()

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): [B, M, C, H, W]
        """
        mean = self.norm_config['mean']
        std = self.norm_config['std']
        
        img = deepcopy(img)
        B, M, C, H, W = img.size()
        img = img.permute(0, 1, 3, 4, 2) # [B, M, C, H, W] => [B, M, H, W, C]
        img = img * torch.tensor(std) + torch.tensor(mean)
        # pixel value [0, 255]
        assert img.min() >= 0 and img.max() <= 255, "Image pixel out of range"

        for b in range(B):
            for m in self.crash_camera:
                img[b, m] = 0

        assert img.min() >= 0 and img.max() <= 255, "Image pixel out of range"
        # img = img - torch.tensor(mean) / torch.tensor(std)
        # img = img.permute(0, 1, 4, 2, 3)

        return img.numpy()

    def get_crash_camera(self):
        crash_camera = np.random.choice([0, 1, 2, 3, 4, 5], size=self.severity)
        return list(crash_camera)


@CORRUPTIONS.register_module()
class FrameLost():
    def __init__(self, severity, norm_config):
        """ 
        Create corruptions: 'Frame Lost'.
        """
        self.severity = severity
        assert severity >= 1 and severity <= 5, f"Corruption Severity should between (1, 5), now {severity}"
        self.norm_config = norm_config
        self.corruption = 'frame_lost'

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): [B, M, C, H, W]
        """
        mean = self.norm_config['mean']
        std = self.norm_config['std']
        
        img = deepcopy(img)
        B, M, C, H, W = img.size()
        img = img.permute(0, 1, 3, 4, 2) # [B, M, C, H, W] => [B, M, H, W, C]
        img = img * torch.tensor(std) + torch.tensor(mean)
        # pixel value [0, 255]
        assert img.min() >= 0 and img.max() <= 255, "Image pixel out of range"

        for b in range(B):
            for m in range(M):
                if np.random.rand() < (self.severity * 1. / 6.):
                    img[b, m] = 0

        assert img.min() >= 0 and img.max() <= 255, "Image pixel out of range"
        # img = img - torch.tensor(mean) / torch.tensor(std)
        # img = img.permute(0, 1, 4, 2, 3)

        return img.numpy()


@CORRUPTIONS.register_module()
class ColorQuant(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Color Quantization'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'color_quant')

    def _get_corrupt_func(self):
        return functools.partial(self.color_quant, severity=self.severity)

    def color_quant(self, x, severity):
        bits = 5 - severity
        x = Image.fromarray(np.uint8(x))
        x = PIL.ImageOps.posterize(x, bits)
        return np.asarray(x)


@CORRUPTIONS.register_module()
class Pixlate(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Pixelate'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'pixelate')


@CORRUPTIONS.register_module()
class Saturate(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Saturate'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'saturate')

    def _get_corrupt_func(self):
        return functools.partial(self.saturate, severity=self.severity)

    def saturate(self, x, severity):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

        x = np.array(x) / 255.
        x = rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = hsv2rgb(x)

        return np.clip(x, 0, 1) * 255
    


@CORRUPTIONS.register_module()
class Rain(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Rain'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'rain')

    def _get_corrupt_func(self):
        return functools.partial(self.rain, severity=self.severity)

    def rain(self, x, severity):
        # verify_image(image)
        image = np.asarray(x)
        slant = -1
        drop_length = 20
        drop_width = 1
        drop_color = (220, 220, 220)
        rain_type = severity
        darken_coefficient = [0.8, 0.8, 0.7, 0.6, 0.5]
        slant_extreme = slant

        imshape = image.shape
        if slant_extreme == -1:
            slant = np.random.randint(-10, 10)  ##generate random slant if no slant value is given
        rain_drops, drop_length = self.generate_random_lines(imshape, slant, drop_length, rain_type)
        output = self.rain_process(image, slant_extreme, drop_length, drop_color, drop_width, rain_drops,
                            darken_coefficient[severity - 1])
        image_RGB = output

        return image_RGB

    @staticmethod
    def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops, darken):
        imshape = image.shape
        rain_mask = np.zeros((imshape[0], imshape[1]))
        image_t = image.copy()
        for rain_drop in rain_drops:
            cv2.line(rain_mask, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length),
                    drop_color, drop_width)

        rain_mask = np.stack((rain_mask, rain_mask, rain_mask), axis=2)
        image_rain = image + np.array(rain_mask * (1 - image / 255.0) * (1 - np.mean(image) / 255.0), dtype=np.uint8)
        blur_rain = cv2.blur(image_rain, (3, 3))  ## rainy view are blurry
        image_RGB = np.array(blur_rain * rain_mask / 255.0 + image * (1 - rain_mask / 255.0))
        # blur_rain_mask=rain_mask
        image_RGB = np.array(image_RGB) / 255.
        means = np.mean(image_RGB, axis=(0, 1), keepdims=True)
        image_RGB = np.array(np.clip((image_RGB - means) * darken + means, 0, 1) * 255, dtype=np.uint8)

        return image_RGB

    @staticmethod
    def generate_random_lines(imshape, slant, drop_length, rain_type):
        drops = []
        area = imshape[0] * imshape[1]
        no_of_drops = area // 600

        # if rain_type.lower()=='drizzle':

        if rain_type == 1:
            no_of_drops = area // 770
            drop_length = 10
            # print("drizzle")
        # elif rain_type.lower()=='heavy':
        elif rain_type == 2:
            no_of_drops = area // 770
            drop_length = 30
        # elif rain_type.lower()=='torrential':
        elif rain_type == 3:
            no_of_drops = area // 770
            drop_length = 60
            # print("heavy")
        elif rain_type == 4:
            no_of_drops = area // 500
            drop_length = 60
        elif rain_type == 5:
            no_of_drops = area // 400
            drop_length = 80
            # print('torrential')

        for i in range(no_of_drops):  ## If You want heavy rain, try increasing this
            if slant < 0:
                x = np.random.randint(slant, imshape[1])
            else:
                x = np.random.randint(0, imshape[1] - slant)
            y = np.random.randint(0, imshape[0] - drop_length)
            drops.append((x, y))
        return drops, drop_length


@CORRUPTIONS.register_module()
class H256ABRCompression(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'H256ABRCompression'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'h256_abr_compression')

    def __call__(self, src, dst):
        c = [2, 4, 8, 16, 32][self.severity - 1]
        result = subprocess.Popen(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", src],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

        data = json.load(result.stdout)

        bit_rate = str(int(float(data['format']['bit_rate']) / c))

        return_code = subprocess.call(
            ["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize",
            bit_rate, dst])

        return return_code



@CORRUPTIONS.register_module()
class BitError(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'BitError'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'bit_error')

    def __call__(self, src, dst):
        c=[100000, 50000, 30000, 20000, 10000][self.severity-1]
        return_code = subprocess.run(
            ["ffmpeg","-y", "-i", src, "-c", "copy", "-bsf", "noise={}".format(str(c)),
            dst])

        return return_code