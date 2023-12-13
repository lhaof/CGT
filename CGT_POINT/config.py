import importlib
import random

import cv2
import numpy as np
import os
from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "CGT"
        model_mode = "original" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = 4 # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        aug_shape = [256,256] # patch shape used during augmentation (larger patch may have less border artefacts)
        act_shape = [256,256] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [256,256] # patch shape at output of network


        self.dataset_name = "fs" # extracts dataset info from dataset.py
        self.log_dir = "logs/" # where checkpoints will be saved

        # paths to training and validation patches
        self.train_dir_list = [
            "/data1/partitionA/CUHKSZ/histopath_2022/codes/GCN_classification_experiments/BRCA_GCN_edgefeat/BRCA/brca/fs/fs/train/256x256_128x128"
        ]
        self.valid_dir_list = [
            "/data1/partitionA/CUHKSZ/histopath_2022/codes/GCN_classification_experiments/BRCA_GCN_edgefeat/BRCA/brca/fs/fs/valid/256x256_128x128"
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
