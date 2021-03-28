#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/03/27 23:55:30
@author      :Caihao (Chris) Cui
@file        :dataset.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib
from PIL import ImageFile
import torch
from torch.utils.data import Dataset
import logging
import numpy as np
from PIL import Image

import logging
logging.basicConfig(level=logging.INFO,
                    # filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    # format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s-%(levelname)s - %(message)s')
# format='%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

LABEL_number = 23  # enforce the index

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BasicDataset(Dataset):
    def __init__(self, df_data, scale=1):
        self.img_list = df_data['img_rgb'].to_list()
        self.mask_list = df_data['img_mask'].to_list()
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = df_data.index.to_list()
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return "BasicDataset"

    def __str__(self):
        return f"Dataset has {len(self.ids)} imagepairs. Scale: {self.scale}."

    @classmethod
    def preprocess(cls, pil_img, scale, Imagetype):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH), Image.ANTIALIAS)

        img_np = np.array(pil_img, dtype=np.uint8)

        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=2)

        # HWC to CHW
        img_processed = img_np.transpose((2, 0, 1))
        if Imagetype == 'rgb':
            if img_processed.max() > 1.0:
                img_processed = img_processed / 255
        elif Imagetype == 'mask':
            # ! To keep the label index min/max same as the original mask
            if img_processed.max() > LABEL_number-1:
                if img_processed.dtype == np.uint8:
                    img_processed = img_processed.astype(np.float)
                    img_processed = img_processed/img_processed.max() * (LABEL_number-1)
                    img_processed = img_processed.astype(np.uint8)
        else:
            pass
        return img_processed

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = self.mask_list[idx]
        img_file = self.img_list[idx]

        mask = Image.open(mask_file)
        img = Image.open(img_file)

        img = self.preprocess(img, self.scale, 'rgb')
        mask = self.preprocess(mask, self.scale, 'mask')
        return {
            'image': torch.from_numpy(img).type(torch.float),
            'mask': torch.from_numpy(mask).type(torch.long)
        }
