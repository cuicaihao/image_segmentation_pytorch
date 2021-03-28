#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/03/27 23:42:40
@author      :Caihao (Chris) Cui
@file        :review.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib

import shutil
import numpy as np
import sys
from skimage import io
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# issue solved by (OSError: image file is truncated) https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162


import logging
logging.basicConfig(level=logging.INFO,
                    # filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    # format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s-%(levelname)s - %(message)s')
                    # format='%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)


IMAGE_Suffix = "*.jpg"
MASK_Suffix = "*.png"


def get_imagepair_lists(image_dir, mask_dir):
    logger.info('Get Imagepair Lists')
    img_files = get_image_list(image_dir, IMAGE_Suffix)
    mask_files = get_image_list(mask_dir, MASK_Suffix)

    # Additional Checking for missing image name
    assert len(img_files) == len(mask_files), "Image Pairs are missing"
    return img_files, mask_files


def get_image_list(img_dir, img_suffix):
    logger.info(f'Get Imagepair List from {img_dir}/{img_suffix}')
    assert Path(img_dir).is_dir(), f"Images {img_suffix} are missing"
    img_files = list(Path(img_dir).glob(img_suffix))
    return img_files


def get_label_names(labels_path):
    logger.info(f'Get label names from {labels_path}')
    assert Path(labels_path).is_file(), "label.txt is missing"
    with open(str(labels_path), "r", encoding="utf-8") as f:
        label_names = [line.strip() for line in f]
    return label_names


def get_imagepair_paths(img_files, mask_dir, idx):
    img_file_name = img_files[idx]
    mask_file_name = str(img_file_name.stem) + MASK_Suffix[1:]
    mask_file_name = Path(mask_dir) / mask_file_name
    return str(img_file_name), str(mask_file_name)


def get_imagepair_df_basic(img_files, mask_dir):
    logger.info('Create imagepair dataframe and Check labels')
    df_data = pd.DataFrame(columns=[
                           'img_rgb', 'img_mask', 'label_min', 'label_max', 'label_unique', 'label_id_list'])
    total_sample_number = len(img_files)
    with tqdm(total=total_sample_number, file=sys.stdout) as pbar:
        for i in range(total_sample_number):
            pbar.set_description('processed: %d' % (1 + i))
            
            prgb, pmask = get_imagepair_paths(img_files, mask_dir, i)
            # read the imagepair
            img = io.imread(prgb)
            mask = io.imread(pmask)
            # check the image size
            Hi, Wi, Ci = img.shape
            Hm, Wm = mask.shape
            assert Hi==Hm and Wi==Wm, f"image pairs {prgb}:{pmask} do not match in shape."
            # check the label index
            mask_min = np.min(mask)
            mask_max = np.max(mask)
            mask_unique_elements = np.unique(mask)
            mask_unique_number = len(mask_unique_elements)
            one_record = [prgb, pmask, mask_min, mask_max,
                          mask_unique_number, mask_unique_elements]
            df_data.loc[i] = one_record
            pbar.update(1)
    df_data.sort_values(by=['label_unique', 'img_rgb'],
                        ascending=[False, True], inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    return df_data


def get_rgb_image_df_basic(img_files):
    logger.info(f'Create rgb image dataframe')
    df_data = pd.DataFrame(columns=['img_name', 'height', 'width', 'channel'])
    total_sample_number = len(img_files)
    with tqdm(total=total_sample_number, file=sys.stdout) as pbar:
        for i in range(total_sample_number):
            prgb = str(img_files[i])
            img_np = io.imread(prgb)
            H, W, C = img_np.shape
            one_record = [prgb, H, W, C]
            df_data.loc[i] = one_record
            pbar.set_description('processed: %d' % (1 + i))
            pbar.update(1)
    df_data.sort_values(by=['img_name'],
                        ascending=False, inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    return df_data


def save_df(df_data, df_dir, df_name):
    df_file_path = Path(df_dir) / df_name
    logger.info(f'Save dataframe {str(df_file_path)}')
    if df_file_path.exists():  # create a backup before overwrite
        df_backupfile_path = Path(df_dir) / (df_name + '.backup')
        # preserve file metadata.
        shutil.copy2(df_file_path, df_backupfile_path)
    df_csv_file_path =  str(df_file_path)
    df_data.to_csv(df_csv_file_path, index=False)
    return df_csv_file_path


def get_train_valid_split(df_data, df_dir, ratio = 0.75):
    logger.info(f'Split Train/Valid dataframe: {ratio}')
    total_sample_num = df_data.shape[0]
    assert ratio > 0 and ratio <1 , "Wrong Ratio"
    pos = int(total_sample_num * ratio)
    df_train = df_data.loc[:pos-1, ].copy()  
    df_valid = df_data.loc[pos:, ].copy()
    save_df(df_train, df_dir, 'df_train.csv')
    save_df(df_valid, df_dir, 'df_valid.csv')
    return df_train, df_valid


def data_review(image_dir, mask_dir, intrim_data_dir, processed_data_dir, test_dir, ratio = 0.75):

    # read and split data for training and validation
    img_files, mask_files  = get_imagepair_lists(image_dir, mask_dir )
    df_data = get_imagepair_df_basic(img_files, mask_dir)
    save_df(df_data, intrim_data_dir, 'df_data.csv')
    df_train, df_valid = get_train_valid_split(df_data, processed_data_dir, ratio = ratio)

    # read the test data
    img_test_files = get_image_list(test_dir, '*.jpg')
    df_test  = get_rgb_image_df_basic(img_test_files)
    save_df(df_test, processed_data_dir, 'df_test.csv')


    return df_train, df_valid, df_test


def label_review(labels_path, top_label_id, processed_data_dir):
    label_names  = get_label_names(labels_path)
    total_label_number = top_label_id + 1
    unknown_label_number = total_label_number - len(label_names)
    if unknown_label_number > 0:
        logger.info(f'{unknown_label_number} label names are missing!')
    for i in range(unknown_label_number):
        unknown_label_name = "unknown_" + str(i+1)
        label_names.append(unknown_label_name)
    label_id = list(range(total_label_number))
 
    assert len(label_id) == len(label_names), "Label ids missmatch names!"

    df_label = pd.DataFrame({'label_id':label_id, "label_name": label_names})
    df_label= df_label.set_index('label_id')
    save_df(df_label, processed_data_dir, 'df_label.csv')
    return df_label
