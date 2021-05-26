import argparse
import os, sys

import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm


def rotate_gt(args, categories_dict, scannet_shape_ids, angles):

    for category in categories_dict:
        cat_path = categories_dict[category] + '_geo'
        cat_save_path = os.path.join(args.data_dir, category + '_geo_8rot')
        os.makedirs(cat_save_path, exist_ok=True)
        for file in tqdm(os.listdir(cat_path)):
            partnet_id = file.split('.')[0]
            if file.endswith('.npy') and partnet_id in scannet_shape_ids:
                shape = np.load(os.path.join(cat_path, file))
                num_parts = len(shape)
                for k, angle in enumerate(angles):
                    rotated_parts = []
                    for i in range(num_parts):
                        part = shape[i, 0, ...]
                        rotated_part = rotate(part, angle, axes=[0, 2], reshape=False)
                        rotated_parts += [rotated_part[None, ...]]
                    rotated_parts = np.stack(rotated_parts)
                    np.save(os.path.join(cat_save_path, f'{partnet_id}_{k}.npy'), rotated_parts)
                full_shape = np.load(os.path.join(cat_path, partnet_id + '_full.npy'))[0]
                for k, angle in enumerate(angles):
                    rotated_shape = rotate(full_shape, angle, axes=[0, 2], reshape=False)[None, ...]
                    np.save(os.path.join(cat_save_path, f'{partnet_id}_full_{k}.npy'), rotated_shape)


def rotate_crops(args, categories_dict, scannet_shape_ids, angles, scannet_train, scannet_val):

    suffix_data = '_scannet_geo'

    for category in categories_dict:
        cat_path = categories_dict[category] + suffix_data
        cat_save_path = os.path.join(args.data_dir, category + suffix_data + '_8rot')
        os.makedirs(cat_save_path, exist_ok=True)
        for file in tqdm(os.listdir(cat_path)):
            if file.endswith('scan.npy') or file.endswith('labels.npy') \
                    or file.endswith('min_1.npy') or file.endswith('max_1.npy') or file.endswith('max_2.npy') \
                    or file.endswith('iou.npy'):
                continue
            partnet_id = file.split('_')[0]
            if file.endswith('.npy') and partnet_id in scannet_shape_ids:
                scannet_id = file.split('_')[1] + '_' + file.split('_')[2]
                filename = file.split('.')[0]
                shape = np.load(os.path.join(cat_path, file))
                for k, angle in enumerate(angles):
                    rotated_shape = rotate(shape, angle, axes=[0, 2], reshape=False)
                    np.save(os.path.join(cat_save_path, f'{filename}_{k}.npy'), rotated_shape)

    for category in categories_dict:
        train_ids = []
        val_ids = []
        cat_path = categories_dict[category] + suffix_data + '_8rot'
        for file in tqdm(os.listdir(cat_path)):
            partnet_id = file.split('_')[0]
            if file.endswith('.npy') and partnet_id in scannet_shape_ids:
                scannet_id = file.split('_')[1] + '_' + file.split('_')[2]
                if scannet_id in scannet_train:
                    train_ids += [file.split('.')[0]]
                if scannet_id in scannet_val:
                    val_ids += [file.split('.')[0]]
        with open(os.path.join(cat_path, 'train.txt'), 'w') as f:
            for item in train_ids:
                f.write("%s\n" % item)
        with open(os.path.join(cat_path, 'val.txt'), 'w') as f:
            for item in val_ids:
                f.write("%s\n" % item)
        with open(os.path.join(cat_path, 'full.txt'), 'w') as f:
            for item in train_ids + val_ids:
                f.write("%s\n" % item)


def rotate_priors(args):
    priors_save_path = args.all_priors_dir + '_8rot'
    os.makedirs(priors_save_path, exist_ok=True)
    for prior_path in os.listdir(args.all_priors_dir):
        prior_name = prior_path.split('.')[0]
        priors = np.load(os.path.join(args.all_priors_dir, prior_path))
        num_priors = len(priors)
        for k, angle in enumerate(angles):
            rotated_priors = []
            for i in range(num_priors):
                prior = priors[i]
                rotated_prior = rotate(prior, angle, axes=[0, 2], reshape=False)
                rotated_priors += [rotated_prior]
            rotated_priors = np.stack(rotated_priors)
            np.save(os.path.join(priors_save_path, f'{prior_name}_{k}.npy'), rotated_priors)



if __name__ == '__main__':

    # params
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('--save_dir', required=True, help='path to store scan data specs')
    parser.add_argument('--data_dir', required=True, help='path to directory with processed trees and scan crops')
    parser.add_argument('--all_data_dir', required=True, help='path to directory with scan data specs')
    parser.add_argument('--all_priors_dir', required=True, help='path to directory with gathered priors data')
    parser.add_argument('--scannet_splits_dir', required=True, help='path to directory with ScanNet splits')

    args = parser.parse_args()

    categories_dict = {}
    categories_dict['chair'] = os.path.join(args.data_dir, 'chair')
    categories_dict['table'] = os.path.join(args.data_dir, 'table')
    categories_dict['storagefurniture'] = os.path.join(args.data_dir, 'storagefurniture')
    categories_dict['bed'] = os.path.join(args.data_dir, 'bed')
    categories_dict['trashcan'] = os.path.join(args.data_dir, 'trashcan')

    scannet_shape_ids = []
    for split in ['train', 'val']:
        with open(os.path.join(args.all_data_dir, split + '.txt'), 'r') as fin:
            lines = fin.readlines()
            lines = [x.split('_')[0] for x in lines]
            scannet_shape_ids += lines
    scannet_shape_ids = list(set(scannet_shape_ids))

    angles = [45 * i for i in range(8)]

    scannet_train = []
    with open(os.path.join(args.scannet_splits_dir, 'scannetv2_train.txt'), 'r') as fin:
        lines = fin.readlines()
        scannet_train = [x[:-1] for x in lines]
    scannet_val = []
    with open(os.path.join(args.scannet_splits_dir, 'scannetv2_val.txt'), 'r') as fin:
        lines = fin.readlines()
        scannet_val = [x[:-1] for x in lines]
    scannet_test = []
    with open(os.path.join(args.scannet_splits_dir, 'scannetv2_test.txt'), 'r') as fin:
        lines = fin.readlines()
        scannet_test = [x[:-1] for x in lines]

    # rotate voxelized GT trees
    rotate_gt(args, categories_dict, scannet_shape_ids, angles)
    # rotate ScanNet crops
    rotate_crops(args, categories_dict, scannet_shape_ids, angles, scannet_train, scannet_val)
    # rotate priors from args.all_priors_dir directory
    rotate_priors(args)
