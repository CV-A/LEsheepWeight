# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os.path as osp

from ..builder import PIPELINES
from PIL import Image

@PIPELINES.register_module()
class LoadImageDepthFromFile:
    """Loading image(s) from file.

    Required key: "image_file".

    Added key: "img".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=True,
                 load_depth=True,
                 color_type='color',
                 channel_order='rgb',
                 gray=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.load_depth = load_depth
        self.color_type = color_type
        self.channel_order = channel_order
        self.gray = gray
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Loading image(s) from file."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        image_file = results['image_file']

        def load_single_file(image_file):
            img_bytes = self.file_client.get(image_file)
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.color_type,
                channel_order=self.channel_order)
            if self.to_float32:
                img = img.astype(np.float32)
            if self.gray:
                img = mmcv.rgb2gray(img, True)
            if img is None:
                raise ValueError(f'Fail to read {image_file}')

            if self.load_depth:
                depth_image_file = image_file.replace("color", "depth").replace("jpg", "tif")
                if osp.exists(depth_image_file):
                    dep_bytes = self.file_client.get(depth_image_file)
                    dep = mmcv.imfrombytes(
                        dep_bytes,
                        flag="unchanged",
                        channel_order=self.channel_order)
                    if self.to_float32:
                        dep = dep.astype(np.float32)
                    if dep is None:
                        raise ValueError(f'Fail to read {image_file}')
                else:
                    dep = np.zeros(img.shape[:-1], dtype=np.float32)
                img = np.concatenate((img, dep[:, :, None]), -1)

            return img

        if isinstance(image_file, (list, tuple)):
            imgs = []
            for image in image_file:
                img = load_single_file(image)
                imgs.append(img)
            results['img'] = imgs
        else:
            img = load_single_file(image_file)
            results['img'] = img

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
