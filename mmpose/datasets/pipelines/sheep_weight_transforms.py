#!/usr/bin/env python3
from mmpose.models.utils.ops import resize
from mmcv import imresize
from mmcv.parallel import DataContainer as DC
from ..builder import PIPELINES
from torchvision.transforms import functional as F
import torchvision
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomGrayscale,
    RandomRotation,
    GaussianBlur,
)


@PIPELINES.register_module()
class SheepRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.funer = RandomHorizontalFlip(p)
    def __call__(self, results):
        results['img'] = self.funer(results['img'])
        return results

@PIPELINES.register_module()
class SheepRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.funer = RandomVerticalFlip(p)
    def __call__(self, results):
        results['img'] = self.funer(results['img'])
        return results

@PIPELINES.register_module()
class SheepRandomGrayscale:
    def __init__(self, p=0.1):
        self.funer = RandomGrayscale(p)
    def __call__(self, results):
        results['img'] = self.funer(results['img'])
        return results

@PIPELINES.register_module()
class SheepRandomRotation:
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        self.funer = RandomRotation(degrees, resample=False, expand=False, center=None, fill=None)
    def __call__(self, results):
        results['img'] = self.funer(results['img'])
        return results

@PIPELINES.register_module()
class SheepGaussianBlur:
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.funer = GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    def __call__(self, results):
        results['img'] = self.funer(results['img'])
        return results


@PIPELINES.register_module()
class SheepResize:
    def __init__(self, size, interpolation=2):
        self.funer = torchvision.transforms.Resize(size, interpolation)
    def __call__(self, results):
        results['img'] = self.funer(results['img'])
        return results



@PIPELINES.register_module()
class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, results):
        # self.size.extend(results['img'].size()[-1])
        # print(self.size, results['img'].shape)
        results['img'] = imresize(results['img'],
                                  self.size,
                                  interpolation='nearest',
                                  return_scale=False,
                                  backend='cv2')
        return results


@PIPELINES.register_module()
class SheepCollect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in `keys` as it is, and collect items in `meta_keys`
    into a meta item called `meta_name`.This is usually the last stage of the
    data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str|tuple]): Required keys to be collected. If a tuple
          (key, key_new) is given as an element, the item retrieved by key will
          be renamed as key_new in collected data.
        meta_name (str): The name of the key that contains meta information.
          This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str|tuple]): Keys that are collected under
          meta_name. The contents of the `meta_name` dictionary depends
          on `meta_keys`.
    """

    def __init__(self, keys, meta_keys, meta_name='img_metas'):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
              to the next transform in pipeline.
        """
        if 'ann_info' in results:
            results.update(results['ann_info'])

        data = {}
        for key in self.keys:
            if isinstance(key, tuple):
                assert len(key) == 2
                key_src, key_tgt = key[:2]
            else:
                key_src = key_tgt = key
            data[key_tgt] = results[key_src]

        meta = {"bbox_id":"FOO"}
        if len(self.meta_keys) != 0:
            for key in self.meta_keys:
                if isinstance(key, tuple):
                    assert len(key) == 2
                    key_src, key_tgt = key[:2]
                else:
                    key_src = key_tgt = key
                meta[key_tgt] = results[key_src]
        data[self.meta_name] = DC(meta, cpu_only=True)

        return data

    def __repr__(self):
        """Compute the string representation."""
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')
