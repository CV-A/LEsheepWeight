#!/usr/bin/env python3
import numpy as np
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only

@HOOKS.register_module()
class ResetCheckpointBestScore(Hook):

    def __init__(self, init_value=np.inf):
        self._init_value = np.float32(init_value)

    @master_only
    def before_run(self, runner):
        if hasattr(runner.meta, 'hook_msgs'):
            runner.meta['hook_msgs']['best_score'] = self._init_value
