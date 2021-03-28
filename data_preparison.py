#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/03/28 18:37:31
@author      :Caihao (Chris) Cui
@file        :data_preparison.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib

from SegNet.data.review import data_review, label_review

# data dir / types
raw_data_dir =  "./data/raw"
intrim_data_dir =  "./data/interim"
processed_data_dir = "./data/processed"

# images to build model
image_dir = raw_data_dir +"/train/images"
mask_dir = raw_data_dir + "/train/masks"
labels_path = raw_data_dir+"/labels.txt"

# image to test model without mask
test_dir = "data/raw/test/images"

# review data 
df_train, df_valid, df_test = data_review(image_dir, mask_dir, intrim_data_dir, processed_data_dir, test_dir)

# review labels.
top_label_id = df_train['label_max'].max()
df_label  = label_review(labels_path, top_label_id, processed_data_dir)
label_names = df_label['label_name'].to_list()