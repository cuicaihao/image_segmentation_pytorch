
from SegNet.features.build_features import *
import numpy as np

def test_features():
    print("This is a test function for SegNet.features")
    return True


def test_2d_to_3d():
    input = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    target = np.array([ [[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]]  ] , dtype=np.uint8)
    output = convert_Mask_Onehot(input, 3)
    assert np.array_equal(output, target), "True if two arrays have the same shape and elements, False otherwise."


def test_dice_coef():
    # inputs
    y_true = np.array([[1, 0], [0, 0]])
    y_pred = np.array([[1, 0], [1, 0]])
    target = 2.0 / 3.0 # dice 
    output = dice_coef(y_true, y_pred)
    assert abs(target - output) <= 1e-5

def test_dice_coeff_multilabel():
    # input
    numLabels = 3
    y_true = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    y_true =  convert_Mask_Onehot(y_true, numLabels)
    #        np.array([ [[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]]  ] 
    y_pred = np.array([ [[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]]  ] , dtype=np.uint8)
    y_true = np.expand_dims(y_true, axis = 0)
    y_pred = np.expand_dims(y_pred, axis = 0)
    assert len(y_true.shape) == len(y_pred.shape)
    target = 1.0
    output =  dice_coef(y_pred, y_true)
    assert abs(target - output) <= 1e-5