#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/03/28 13:44:01
@author      :Caihao (Chris) Cui
@file        :build_features.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib

import numpy as np 


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels): # N C H W
    dice=0
    for c in range(numLabels):
        dice += dice_coef(y_true[:,c,:,:], y_pred[:,c,:,:])
    return dice/numLabels # taking average

def metricComputation(y_true, y_pred):
    # A is ground truth.
    # B is the prediction from the model.
    A = y_true.astype(np.float32)
    B = y_pred.astype(np.float32)

    # Evaluate TP, TN, FP, FN
    SumAB = A + B
    minValue = np.min(SumAB)
    maxValue = np.max(SumAB)

    TP = len(SumAB[np.where(SumAB == maxValue)])
    TN = len(SumAB[np.where(SumAB == minValue)])

    SubAB = A - B
    minValue = np.min(SubAB)
    maxValue = np.max(SubAB)
    FP = len(SubAB[np.where(SubAB == minValue)])
    FN = len(SubAB[np.where(SubAB == maxValue)])

    Accuracy = (TP+TN)/(FN+FP+TP+TN)
    Precision = TP/(TP+FP)
    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    Fmeasure = 2*TP/(2*TP+FP+FN)

    MCC = (TP*TN-FP*FN)/np.sqrt(float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    Dice = 2*TP/(2*TP+FP+FN)
    Jaccard = Dice/(2-Dice)

    scores = {}
    scores["Accuracy"] = Accuracy
    scores["Sensitivity"] = Sensitivity
    scores["Precision"] = Precision
    scores["Specificity"] = Specificity
    scores["Fmeasure"] = Fmeasure
    scores["MCC"] = MCC
    scores["Dice"] = Dice
    scores["IoU (Jacard)"] = Jaccard
    print("="*64)
    print("[Metric Computation] ")
    for k, v in scores.items():
        print(f"{k:15}=> {v:10f}")
    print("-"*64)
    return scores

 