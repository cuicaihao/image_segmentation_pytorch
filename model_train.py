#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/03/28 00:14:58
@author      :Caihao (Chris) Cui
@file        :train_model.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from SegNet.models.unet import UNet, UNetSmall
from SegNet.data.dataset import BasicDataset
from SegNet.models.dice_loss import dice_coeff
# from SegNet.models.dice_loss import  CrossEntropyLoss2d

from SegNet.features.build_features import dice_coeff_multilabel

import pandas as pd

from torch.utils.tensorboard import SummaryWriter

import logging

logging.basicConfig(level=logging.INFO,
                    # filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    # format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s-%(levelname)s - %(message)s')
# format='%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)


df_train_path = "data/processed/df_train.csv"
df_valid_path = "data/processed/df_valid.csv"

dir_checkpoint = 'checkpoints/'

LABEL_number = 23

# DEBUG = True
DEBUG = False

def train_net(net,
              device,
              epochs=5,
              batch_size=4,
              lr=0.001,
              save_cp=True,
              img_scale=0.1):

    # # Full large dataset
    df_train = pd.read_csv(df_train_path)
    df_valid = pd.read_csv(df_valid_path)
    if not DEBUG:
        data_train = BasicDataset(df_train, scale = img_scale)
        data_valid = BasicDataset(df_valid, scale = img_scale )
    else:
    # Small dataset for debugging and test
        df_train_small = df_train.head(10)
        df_valid_small = df_valid.head(2)
        data_train = BasicDataset(df_train_small, scale=img_scale)
        data_valid = BasicDataset(df_valid_small, scale=img_scale)

    train_loader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)

    writer = SummaryWriter(
        comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    n_train = len(data_train)
    n_val = len(data_valid)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    optimizer = optim.RMSprop(net.parameters(), lr=lr,
                              weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
        # criterion = CrossEntropyLoss2d()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                true_masks_temp = torch.squeeze(true_masks, dim=1)

                masks_pred = net(imgs)
                masks_pred = F.softmax(masks_pred, dim=1)

                # masks_pred = torch.argmax(masks_pred, dim=1)
                loss = criterion(masks_pred, true_masks_temp)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (n_train // (2 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram(
                            'weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram(
                            'grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    # validating the net
                    val_score = eval_net(net, val_loader, device)
                    # update learning rate
                    scheduler.step(val_score)
                    writer.add_scalar(
                        'learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    if net.n_classes > 1:
                        # logging.info('Validation cross entropy: {}'.format(val_score))
                        logging.info('Validation Dice Coeff multilabel: {}'.format(val_score))
                        writer.add_scalar('Loss/val', val_score, global_step)
                    else:
                        logging.info(
                            'Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/val', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images(
                            'masks/true', true_masks, global_step)
                        writer.add_images(
                            'masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                    else:
                        writer.add_images(
                            'masks/true', true_masks, global_step)
                        masks_pred_temp = torch.argmax(masks_pred, dim=1, keepdim=True).type(torch.long)  # same as true_masks
                        writer.add_images(  
                            'masks/pred', masks_pred_temp, global_step)          

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.add_graph(net, imgs)
    writer.close()



def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                mask_pred = F.softmax(mask_pred, dim=1)

            if net.n_classes > 1:
                true_masks = torch.squeeze(true_masks, dim=1)
                # tot += F.cross_entropy(mask_pred, true_masks).item()
                tot += dice_coeff_multilabel(true_masks, mask_pred, LABEL_number)
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()
    net.train()
    return tot / n_val

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default = 2,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.1,
                        help='Downscaling factor of the images')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNetSmall(n_channels=3, n_classes=LABEL_number, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

# python model_train.py -f checkpoints/CP_epoch2.pth
