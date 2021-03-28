#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/03/28 21:30:28
@author      :Caihao (Chris) Cui
@file        :test_models.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib
import numpy as np
import torch
from SegNet.models.dice_loss import dice_coeff
from SegNet.features.build_features  import dice_coeff_multilabel
def test_models():
    # TO-DO: Test SegNet.models
    return True

def test_dice():
    # dummy input
    y_true = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    y_pred = np.array([ [[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]]  ] , dtype=np.uint8)
    y_true = y_true[np.newaxis, :] 
    y_pred = np.expand_dims(y_pred, axis = 0) # N C H W 
    assert len(y_true.shape) +1 == len(y_pred.shape)
    
    target = 1.0
    y_true = torch.from_numpy(y_true) 
    y_pred = torch.from_numpy(y_pred) 

    output = dice_coeff_multilabel(y_true, y_pred , numLabels = 3)
    assert abs(target - output) <= 1e-5