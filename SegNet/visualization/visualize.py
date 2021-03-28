#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/03/27 23:42:48
@author      :Caihao (Chris) Cui
@file        :visualize.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

from PIL import Image
from pathlib import Path

# here put the import lib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import logging
logging.basicConfig(level=logging.INFO,
                    # filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    # format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s-%(levelname)s - %(message)s')
# format='%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

IMAGE_pairsize = (20, 10)  # W, H
IMAGE_overlaysize = (18, 12) # W, H

def pil_read_rgb(image_file_name):
    """ RGB (3x8-bit pixels, true color) """
    img = Image.open(image_file_name).convert('RGB')
    # img = Image.open(image_file_name)
    return img


def pil_read_mask(image_file_name):
    """ L (8-bit pixels, black and white) """
    img = Image.open(image_file_name).convert('L')
    # img = Image.open(image_file_name)
    return img


def pil_read(img_file, mask_file):
    img = pil_read_rgb(img_file)
    mask = pil_read_mask(mask_file)
    return img, mask


def preprocess(pil_img, scale=1):
    """convert pillow image to np array"""
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

#     # HWC to CHW
#     img_trans = img_nd.transpose((2, 0, 1))
#     if img_trans.max() > 1:
#         img_trans = img_trans / 255
    return img_nd


def plot_image_pair(pil_img, pil_mask,label_names, cmap='gray', cbar = False, save_dir='reports/figures'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=IMAGE_pairsize)
    ax1.set_title('Input: RGB')
    ax1.imshow(pil_img)
    im = ax2.imshow(pil_mask,  cmap=plt.cm.get_cmap(
        cmap, len(label_names)),  interpolation='none')
    ax2.set_title('Output: MASK')
    if cbar:
        fig.colorbar(im , ax = ax2)
    save_imagepair_path = save_dir + '/image_pair_'+cmap+'.png'
    if save_dir:
        plt.savefig(save_imagepair_path,
                    dpi=200, bbox_inches='tight', pad_inches=0)
        logger.info(f"Save imagepair at {save_imagepair_path}")
    else:
        logger.info(f"No save imagepair.")
    plt.show()
    return save_imagepair_path


def plot_imageoverlay(pil_img, pil_mask, mask_names, cmap='gray', alpha=0.4, save_dir='reports/figures'):
    # img_np = np.array(pil_img)
    # mask_np = np.array(pil_mask)
    N_cmap = len(mask_names)
    name_index = list(range (N_cmap))
    fig, ax = plt.subplots(figsize=IMAGE_overlaysize)  # W, H
    ax.imshow(pil_img)
    im = ax.imshow(pil_mask,  cmap=plt.cm.get_cmap(
        cmap, N_cmap), alpha=alpha,  interpolation='none')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, ticks=name_index)
    cbar.ax.set_yticklabels(mask_names)  # horizontal colorbar
    plt.tight_layout()
    save_imagepair_path = save_dir + '/sample.png'
    if save_dir:
        plt.savefig(save_imagepair_path, dpi=200)
        logger.info(f"Save imageoverlay at {save_imagepair_path}")
    
    plt.show()
    return save_imagepair_path
