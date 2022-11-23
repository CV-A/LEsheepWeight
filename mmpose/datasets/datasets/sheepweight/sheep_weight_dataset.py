#!/usr/bin/env python3

import os.path as osp
import numpy as np
import pandas as pd
import itertools

from torch.utils.data import Dataset
from mmcv import scandir
from mmpose.datasets.pipelines import Compose
from ...builder import DATASETS
from pprint import pprint
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

@DATASETS.register_module()
class SheepWeightDataset(Dataset):
    def __init__(self,
                 img_prefix,
                 csv_file,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):
        self.img_prefix = img_prefix

        self.csv_file = csv_file
        self.csv_df = pd.read_csv(self.csv_file)
        self.csv_df['samplenum'] = self.csv_df.filepath

        self.pipeline = Compose(pipeline)

        ## Remove no weight records
        count = 0
        coco = []
        coco_images = scandir(dir_path=img_prefix,
                              suffix="jpg",
                              recursive=True)
        weighted_samples = set(self.csv_df.samplenum)
        for img in coco_images:
            file_name = osp.join(self.img_prefix, img)
            basename = osp.basename(file_name)
            samplenum = "_".join(basename.split("_")[1:3])
            if samplenum in weighted_samples:
                count += 1
                weight = self.csv_df.weight[ self.csv_df.samplenum == samplenum  ]
                coco.append({
                    'file_name':file_name,
                    'image_file':file_name,
                    'target': np.float32(weight)
                })
        self.coco = coco

    def __len__(self):
        return len(self.coco)

    def __getitem__(self,idx):
        results = self.pipeline(self.coco[idx])
        return results

    def evaluate(self, results, logger=None, **kwargs):

        total_target = []
        total_error = []
        total_errorp = []
        total_output = []
        count = 0

        if hasattr(logger, 'handlers'):
            baseFilename = osp.dirname(logger.handlers[1].baseFilename)
        else:
            baseFilename = "./"
        dflist = []
        alloutputs = itertools.chain(results)
        for batch in alloutputs:
            for target, output, error, img_metas in zip(
                batch["target"].tolist(),
                batch["output"].tolist(),
                batch["error"].tolist(),
                batch["img_metas"]
            ):
                dflist.append((img_metas["image_file"], target, output, error))
        df = pd.DataFrame(dflist)
        df = df.sort_values(by=3, key=lambda x: x.abs())
        df.to_csv(
            osp.join(
                baseFilename,
                "evaluation_all_samples_{}.csv".format(
                    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                ),
            ),
            header=["imagefile", "target", "output", "error"],
            index=False
        )


        for batch_data in results:
            error = batch_data["error"]
            target = batch_data['target']
            output = batch_data['output']

            count += error.size
            errorp = error / target

            if isinstance(errorp, np.float32):
                total_target.append(target)
                total_error.append(error)
                total_errorp.append(errorp)
                total_output.append(output)
            else:
                total_output.extend(output.tolist())
                total_target.extend(target.tolist())
                total_error.extend(error.tolist())
                total_errorp.extend(errorp.tolist())

        total_errorp = np.array(total_errorp) * 100
        total_target = np.array(total_target)
        total_output = np.array(total_output)
        total_error = np.array(total_error)

        MSE = mean_squared_error(total_target, total_output)
        RMSE = np.sqrt(MSE)
        MAE = mean_absolute_error(total_target, total_output)
        MAPE = mean_absolute_percentage_error(total_target, total_output) * 100
        R2 = r2_score(total_target, total_output)
        print(' ', flush=True)
        return {"AbsErrorP": np.abs(total_errorp).mean(),
                "ErrorP": total_errorp.mean(),
                "SingleMaxP": np.abs(total_errorp).max(),
                "MSE": MSE,
                "RMSE": RMSE,
                "MAE": MAE,
                "MAPE": MAPE,
                "R2":R2}
                #"IndividualP": total_errorp}
