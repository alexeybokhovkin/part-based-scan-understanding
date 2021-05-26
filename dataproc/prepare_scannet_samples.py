import argparse
import os, sys
import json
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from ..utils.scannet_utils import get_global_part_labels_description, load_pickle, get_scannet
from ..utils.vox import load_sample
from ..utils.transforms import apply_transform, apply_inverse_transform
import main_directories as md


def load_metadata(args):

    global_labels = get_global_part_labels_description(md.DICTIONARIES)
    partnet_to_shapenet_transforms_path = '../dictionaries/partnet_to_shapenet_transforms.pkl'
    partnet_to_shapenet_transforms = load_pickle(partnet_to_shapenet_transforms_path)
    with open(os.path.join(md.DICTIONARIES, 'full_annotations.json'), 'rb') as f:
        scan2cad_anno = json.load(f)

    VALID_PARTNET_IDS = os.path.join(args.trees_dir, f'{args.category}_hier/full.txt')
    valid_partnet_ids = []
    with open(VALID_PARTNET_IDS, 'r') as f:
        lines = f.readlines()
        for line in lines:
            valid_partnet_ids += [line[:-1]]

    with open(os.path.join(md.DICTIONARIES, 'scannetv2_train.txt'), 'r') as fin:
        lines = fin.readlines()
        scannet_train_scenes = [x[:-1] for x in lines]

    with open(os.path.join(md.DICTIONARIES, 'scannetv2_val.txt'), 'r') as fin:
        lines = fin.readlines()
        scannet_val_scenes = [x[:-1] for x in lines]

    return scan2cad_anno, global_labels, partnet_to_shapenet_transforms, \
           scannet_train_scenes, scannet_val_scenes, valid_partnet_ids


def process_scannet(args, scannet_train_scenes, scannet_val_scenes,
                    scan2cad_anno, global_labels, partnet_to_shapenet_transforms, valid_partnet_ids,
                    save_dir):

    size = args.voxel_dim
    scannet_split = scannet_train_scenes if (args.split == 'train') else scannet_val_scenes

    split_ids = []

    # for each scene at Scan2CAD
    for i, anno_item in enumerate(tqdm(scan2cad_anno)):

        # get Scan2CAD info
        scan_id = anno_item['id_scan']

        if scan_id not in scannet_split:
            continue
        scan_transform = anno_item["trs"]
        aligned_models = anno_item['aligned_models']

        # load scannet point cloud (mesh vertices)
        scan_points_origin = get_scannet(scan_id, args.scannet_dir, output_type='array')

        # transform scan to Scan2CAD coordinate system
        scan_points = apply_transform(scan_points_origin, scan_transform)

        # init_scan_mask
        semantic_mask = -np.ones(len(scan_points))
        instance_mask = -np.ones(len(scan_points))

        # for each aligned shape
        object_id = 0
        for j, anno_shape in enumerate(aligned_models):
            # get Scan2CAD info about shape
            category_id = anno_shape['catid_cad']
            shape_id = anno_shape['id_cad']

            # get_global_info
            df_parts = global_labels[
                (global_labels['category_id'] == category_id) & (global_labels['object_id'] == shape_id)
                ]
            if len(df_parts) == 0:
                continue

            # load shape pointcloud
            partnet_id = df_parts.object_partnet_id.values[0]
            partnet_path = os.path.join(args.partnet_dir, partnet_id, 'point_sample')

            if partnet_id not in valid_partnet_ids:
                continue

            # LOAD partnet modalities
            shape_ply = np.loadtxt(os.path.join(partnet_path, 'pts-10000.pts'), delimiter=' ')[:, :3]
            shape_vox = load_sample(os.path.join(md.PARTNET_VOXELIZED, partnet_id, 'full_vox.df'))

            # LOAD WORLD -> SHAPENET -> PARTNET -> PARTNET_VOX transform
            shape_transform = anno_shape["trs"]
            if partnet_id not in partnet_to_shapenet_transforms:
                print('Transform for Partnet shape ', partnet_id, ' not found')
                continue
            partnet_transform = partnet_to_shapenet_transforms[partnet_id]
            voxel_transform = shape_vox.grid2world

            # MAP scan points: WORLD -> SHAPENET -> PARTNET
            scan_points_aligned = apply_inverse_transform(scan_points, shape_transform, partnet_transform)

            # get instance points from the scan
            # calculate distance
            tree = cKDTree(shape_ply)
            min_dist, min_idx = tree.query(scan_points_aligned)

            # Color
            for is_near, i_nn, i_point in zip(min_dist <= 0.07, min_idx, range(len(scan_points_aligned))):
                if is_near:
                    instance_mask[i_point] = object_id

            instance_points = scan_points_aligned[np.where(instance_mask == object_id)[0]]

            if len(instance_points) > 0:
                # Normalization
                min_1 = deepcopy(instance_points.min(0))
                instance_points -= instance_points.min(0)
                max_1 = deepcopy(instance_points.max() / 0.95)
                instance_points /= instance_points.max() / 0.95
                max_2 = deepcopy(instance_points.max(0) / 2)
                instance_points -= instance_points.max(0) / 2

                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{j}_min_1.npy'), min_1)
                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{j}_max_1.npy'), max_1)
                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{j}_max_2.npy'), max_2)

                # MAP instance points: PARTNET -> PARTNET_VOX
                instance_grid = apply_inverse_transform(instance_points, voxel_transform)
                instance_grid_int = instance_grid.astype('int')
                instance_grid_int = np.maximum(0, np.minimum(size - 1, instance_grid_int))

                # set of indices -> grid
                voxel_points = np.zeros((size, size, size)).astype('uint8')
                voxel_points[instance_grid_int[:, 2], instance_grid_int[:, 1], instance_grid_int[:, 0]] = 1

                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{j}.npy'), voxel_points)
                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{j}_scan.npy'), instance_grid)

                split_ids += [f'{partnet_id}_{scan_id}_{j}']
                object_id += 1

    with open(os.path.join(save_dir, f'{args.split}.txt'), 'w') as f:
        for item in split_ids:
            f.write("%s\n" % item)


if __name__ == '__main__':

    # params
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('--save_dir', required=True, help='path to store data')
    parser.add_argument('--category', required=True, help='shape category to process')
    parser.add_argument('--voxel_dim', type=int, default=32, help='size of voxelized shapes')
    parser.add_argument('--trees_dir', required=True, help='path where trees are stored')
    parser.add_argument('--scannet_dir', required=True, help='path to ScanNet dataset')
    parser.add_argument('--partnet_dir', required=True, help='path to PartNet dataset')
    parser.add_argument('--mode', default='train', help='ScanNet split to process')

    args = parser.parse_args()

    SAVE_DIR = os.path.join(args.save_dir, f'{args.category}_scannet_geo_finaltable')
    os.makedirs(SAVE_DIR, exist_ok=True)

    scan2cad_anno, \
    global_labels, \
    partnet_to_shapenet_transforms, \
    scannet_train_scenes, \
    scannet_val_scenes,\
    valid_partnet_ids = load_metadata(args)

    process_scannet(args, scannet_train_scenes, scannet_val_scenes,
                    scan2cad_anno, global_labels, partnet_to_shapenet_transforms, valid_partnet_ids,
                    SAVE_DIR)
