#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/03/28 09:04:19
@author      :Caihao (Chris) Cui
@file        :model_predict.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd


from SegNet.models.unet import UNet, UNetSmall
from SegNet.data.dataset import BasicDataset
from SegNet.visualization.visualize import plot_image_pair, plot_imageoverlay

import logging

logging.basicConfig(level=logging.INFO,
                    # filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    # format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s-%(levelname)s - %(message)s')
# format='%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)


def predict_img(net,
                full_img,
                device,
                scale_factor= 0.1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, 'rgb'))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.argmax(dim=1, keepdim=True).squeeze(0)
        # torch.argmax(masks_pred, dim=1, keepdim=True)
        # mask = probs.gather(0, index).squeeze(0)

        # tf = transforms.Compose(
        #     [
        #         # transforms.ToPILImage(mode = 'L'),
        #         transforms.Resize(full_img.size[1]),
        #         transforms.ToTensor()
        #     ]
        # )
        # probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask, scale):
    pil_img = Image.fromarray(mask.astype(np.uint8))
    w, h = pil_img.size
    newW, newH = int(w / scale), int(h / scale)
    pil_img = pil_img.resize((newW, newH), Image.ANTIALIAS)
    return pil_img

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")                        
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images')

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--nosave', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--maskthreshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default= 0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default = 0.1)
    return parser.parse_args()

if __name__ == "__main__":

    df_label = pd.read_csv("data/processed/df_label.csv")
    label_names = df_label['label_name'].to_list()

    df_test_path = "data/processed/df_test.csv"
    df_test = pd.read_csv(df_test_path)
    input_file_name = df_test['img_name'][0]

    args = get_args()

    # args.input = "data/raw/train/images/382.jpg"
    # args.output = "output_test.png"
    # args.scaling = 0.1
    # args.model = "checkpoints/CP_epoch2.pth"
    # args.nosave = True # save output

    logging.info(f'''Starting Predict:
        model:          {args.model}
        input:          {args.input}
        output:         {args.output}
        viz:            {args.viz}
        nosave:        {args.nosave}
        maskthreshold: {args.maskthreshold}
        scale:          {args.scale}
    ''')

    # out_files = get_output_filenames(args)
    in_files = args.input
    out_files =  args.output

    net = UNetSmall(n_channels=3, n_classes=len(label_names))

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))
        img = Image.open(fn)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.maskthreshold,
                           device=device)
        if not args.nosave:
            out_fn = out_files[i]
            result = mask_to_image(mask, args.scale)
            result.save(out_files[i])
            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_image_pair(img, result, label_names, cmap='jet', save_dir="reports/predict")
            plot_imageoverlay(img, result, label_names, cmap='jet', save_dir = "reports/predict")

