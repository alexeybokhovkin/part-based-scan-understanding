import argparse
import os, sys
import json
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import trimesh
from scipy.ndimage.morphology import binary_erosion, binary_dilation

from ..utils.scannet_utils import get_global_part_labels_description, load_pickle, get_scannet
from ..utils.vox import load_sample
from ..utils.transforms import apply_transform, apply_inverse_transform
import main_directories as md

mag_factors = {'chair': 1.25,
               'table': 1.35,
               'storagefurniture': 1.25,
               'bed': 1.15,
               'trashcan': 1.4}

struct_cats_to_scannet_cats = {'chair': ['chair'],
                               'table': ['table'],
                               'storagefurniture': ['cabinet', 'bookshelf'],
                               'bed': ['bed', 'sofa'],
                               'trashcan': ['garbagebin']}


def perform_correction(mask, voxel_transform, min_1, max_1, max_2, min_1_pd, max_1_pd, max_2_pd, mag_factor, size):

    mask_coords = np.where(mask > 0)
    mask_coords = np.stack([mask_coords[2], mask_coords[1], mask_coords[0]]).T

    mask_coords = apply_transform(mask_coords, voxel_transform)

    mask_coords += max_2
    mask_coords *= max_1
    mask_coords += min_1

    mask_coords -= min_1_pd
    mask_coords /= max_1_pd
    mask_coords -= max_2_pd

    mask_coords *= mag_factor
    mask_coords = apply_inverse_transform(mask_coords, voxel_transform)

    mask_tensor = np.zeros((32, 32, 32))
    mask_coords = mask_coords.astype(int)
    mask_coords = np.maximum(0, np.minimum(size - 1, mask_coords))
    mask_tensor[mask_coords[:, 2], mask_coords[:, 1], mask_coords[:, 0]] = 1

    mask_tensor = binary_dilation(mask_tensor, structure=np.ones((2, 2, 2)), iterations=2)
    mask_tensor = binary_erosion(mask_tensor, structure=np.ones((2, 2, 2)), iterations=2)
    mask_tensor = mask_tensor.astype(int)

    return mask_tensor


def perform_correction_on_parts(mask, voxel_transform, min_1, max_1, max_2, min_1_pd, max_1_pd, max_2_pd, mag_factor, size):

    mask_tensors = []
    all_ratios = []
    for part_mask in mask:
        part_mask = np.squeeze(part_mask)
        mask_coords = np.where(part_mask > 0)
        mask_coords = np.stack([mask_coords[2], mask_coords[1], mask_coords[0]]).T

        mask_coords = apply_transform(mask_coords, voxel_transform)

        mask_coords += max_2
        mask_coords *= max_1
        mask_coords += min_1

        mask_coords -= min_1_pd
        mask_coords /= max_1_pd
        mask_coords -= max_2_pd

        mask_coords *= mag_factor
        mask_coords = apply_inverse_transform(mask_coords, voxel_transform)

        mask_tensor = np.zeros((32, 32, 32))
        mask_coords = mask_coords.astype(int)

        mask_coords_above_limits = [x[0] < 0 or x[0] >= 32 or x[1] < 0 or x[1] >= 32 or x[2] < 0 or x[2] >= 32 for x in
                                    mask_coords]
        above_limits_ratio = sum(mask_coords_above_limits) / mask_coords.shape[0]
        all_ratios += [above_limits_ratio]

        mask_coords = np.maximum(0, np.minimum(size - 1, mask_coords))
        mask_tensor[mask_coords[:, 2], mask_coords[:, 1], mask_coords[:, 0]] = 1

        mask_tensor = binary_dilation(mask_tensor, structure=np.ones((2, 2, 2)), iterations=2)
        mask_tensor = binary_erosion(mask_tensor, structure=np.ones((2, 2, 2)), iterations=2)
        mask_tensor = mask_tensor.astype(int)
        mask_tensors += [mask_tensor[None, ...]]
    mask_tensors = np.stack(mask_tensors)

    return mask_tensors, max(all_ratios)


def iou(mask_1, mask_2):
    smooth = 1e-4
    intersection = mask_1 * mask_2
    union = mask_1 + mask_2 - intersection
    metric = intersection.sum() / (union.sum() + smooth)
    return metric


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

    mlcvnet_bboxes_filenames = []
    for file in os.listdir(args.mlcvnet_output):
        if file.endswith('.ply'):
            mlcvnet_bboxes_filenames += [file]

    global_id_to_semantic_class = load_pickle(os.path.join(md.DICTIONARIES, 'global_id_to_semantic_class.pkl'))

    return scan2cad_anno, global_labels, partnet_to_shapenet_transforms, \
           scannet_train_scenes, scannet_val_scenes, valid_partnet_ids, \
           mlcvnet_bboxes_filenames, global_id_to_semantic_class


def process_mlcvnet(args, scannet_train_scenes, scannet_val_scenes,
                    scan2cad_anno, global_labels, partnet_to_shapenet_transforms, valid_partnet_ids,
                    global_id_to_semantic_class, save_dir):

    size = args.voxel_dim
    scannet_split = scannet_train_scenes if (args.split == 'train') else scannet_val_scenes

    split_ids = []

    type2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
                  'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                  'refrigerator': 12, 'showercurtrain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'garbagebin': 17}
    class2type = {u: v for v, u in type2class.items()}

    # for each scene at Scan2CAD
    for i, anno_item in enumerate(tqdm(scan2cad_anno)):
        all_scene_correspondences = []

        # get Scan2CAD info
        scan_id = anno_item['id_scan']

        if scan_id not in scannet_split:
            continue
        scan_transform = anno_item["trs"]
        aligned_models = anno_item['aligned_models']

        # load raw scannet data
        scan_path = os.path.join(args.scannet_dir, scan_id, scan_id + '_vh_clean_2.ply')
        scan_data = trimesh.load_mesh(scan_path).metadata['ply_raw']['vertex']['data']

        # get scannet point cloud (mesh vertices) and their colors
        scan_points_origin = np.array([list(x) for x in scan_data[['x', 'y', 'z']]])
        scan_color = np.array([list(x) for x in scan_data[['red', 'green', 'blue']]]) / 255

        # transform scan to Scan2CAD coordinate system
        scan_points = apply_transform(scan_points_origin, scan_transform)

        meta_file = os.path.join(args.scannet_dir, scan_id, scan_id + '.txt')
        lines = open(meta_file).readlines()
        axis_align_matrix = None
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
        assert axis_align_matrix is not None
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

        all_instances = []
        all_instances_world = []
        all_instances_partnet = []
        all_scans_aligned = []
        all_partnet_labels = []
        instance_ids = []
        shape_plys = []
        partnet_ids = []
        category_ids = []
        shape_ids = []
        voxel_transforms = []
        shape_transforms = []
        partnet_transforms = []

        all_buf = []

        # for each aligned shape
        object_id = 0
        for j, anno_shape in enumerate(aligned_models):
            # init_scan_mask
            semantic_mask = -np.ones(len(scan_points))
            instance_mask = -np.ones(len(scan_points))

            # get Scan2CAD info about shape
            category_id = anno_shape['catid_cad']
            shape_id = anno_shape['id_cad']

            # get_global_info
            df_parts = global_labels[
                (global_labels['category_id'] == category_id) & (global_labels['object_id'] == shape_id)
                ]
            from_part_id_2_global_id = dict(df_parts.reset_index()[['part_id', 'global_id']].values)
            if len(df_parts) == 0:
                continue

            # load shape pointcloud
            partnet_id = df_parts.object_partnet_id.values[0]
            partnet_path = os.path.join(args.partnet_dir, partnet_id, 'point_sample')

            if partnet_id not in valid_partnet_ids:
                continue

            # load shape part labels
            shape_label = np.loadtxt(os.path.join(partnet_path, 'label-10000.txt'), delimiter='\n')
            shape_label = np.array([from_part_id_2_global_id[p_id] for p_id in shape_label])

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
            instance_points_world = apply_transform(instance_points, partnet_transform, shape_transform)

            if len(instance_points) > 0:
                # Normalization
                instance_points -= instance_points.min(0)
                instance_points /= instance_points.max() / 0.95
                instance_points -= instance_points.max(0) / 2

                # MAP instance points: PARTNET -> PARTNET_VOX
                instance_grid = apply_inverse_transform(instance_points, voxel_transform).astype('int')
                instance_grid = np.maximum(0, np.minimum(size - 1, instance_grid))

                # store data for further processing
                all_instances += [instance_grid]
                all_instances_world += [instance_points_world]
                all_instances_partnet += [instance_points]
                all_scans_aligned += [scan_points_aligned]
                all_partnet_labels += [shape_label]
                shape_plys += [shape_ply]
                instance_ids += [j]

                partnet_ids += [partnet_id]
                category_ids += [category_id]
                shape_ids += [shape_id]

                voxel_transforms += [voxel_transform]
                shape_transforms += [shape_transform]
                partnet_transforms += [partnet_transform]

        mask_instances = [x for x in os.listdir(os.path.join(args.mlcvnet_output, scan_id + '_vh_clean_2')) if
                          x.endswith('.ply')]
        mask_classes = [int(x.split('_')[0]) for x in mask_instances]
        mask_class_names = [class2type[x] for x in mask_classes]

        mask_instances = [mask_instances[i] for i in range(len(mask_instances)) if mask_class_names[i] in struct_cats_to_scannet_cats[args.category]]

        box_vertices = []
        box_vertices_fixed = []
        for k in range(len(mask_instances)):
            box_meshes = trimesh.load(os.path.join(args.mlcvnet_output, scan_id + '_vh_clean_2', mask_instances[k])).split()
            for mesh in box_meshes:
                one_box_vertices = deepcopy(np.array(mesh.vertices))
                z_vec = one_box_vertices[1] - one_box_vertices[0]
                offset = np.array([0, 0, 0.5])
                if z_vec[2] < 0:
                    buf = deepcopy(one_box_vertices[1])
                    one_box_vertices[1] = deepcopy(one_box_vertices[0])
                    one_box_vertices[0] = deepcopy(buf)
                z_vec_fixed = one_box_vertices[1] - one_box_vertices[0] + offset
                z_vec = one_box_vertices[1] - one_box_vertices[0]
                y_vec = one_box_vertices[2] - one_box_vertices[0]
                x_vec = one_box_vertices[4] - one_box_vertices[0]
                z_points_fixed = np.linspace(one_box_vertices[0] - offset, one_box_vertices[1], int(round(abs(z_vec_fixed[2]) / 0.05)))[:, 2]
                z_points = np.linspace(one_box_vertices[0], one_box_vertices[1], int(round(abs(z_vec[2]) / 0.05)))[:, 2]
                y_points = np.linspace(one_box_vertices[0], one_box_vertices[2], int(round(abs(y_vec[1]) / 0.05)))[:, 1]
                x_points = np.linspace(one_box_vertices[0], one_box_vertices[4], int(round(abs(x_vec[0]) / 0.05)))[:, 0]

                box_grid = deepcopy(np.meshgrid(x_points, y_points, z_points))
                box_grid_fixed = deepcopy(np.meshgrid(x_points, y_points, z_points_fixed))
                box_vertices += [np.stack(box_grid).reshape((3, -1)).T]
                box_vertices_fixed += [np.stack(box_grid_fixed).reshape((3, -1)).T]
        box_vertices_non_aligned = []
        box_vertices_non_aligned_fixed = []
        for k, vertices in enumerate(box_vertices):
            box_vertices_non_aligned += [
                apply_transform(apply_inverse_transform(vertices, axis_align_matrix), scan_transform)]
            box_vertices_non_aligned_fixed += [
                apply_transform(apply_inverse_transform(box_vertices_fixed[k], axis_align_matrix), scan_transform)]
        for t, gt_points in enumerate(all_instances_world):
            for q, pd_points in enumerate(box_vertices_non_aligned):
                x_mean = (gt_points[:, 0].max() + gt_points[:, 0].min()) / 2
                y_mean = (gt_points[:, 1].max() + gt_points[:, 1].min()) / 2
                z_mean = (gt_points[:, 2].max() + gt_points[:, 2].min()) / 2
                gt_center = np.array([x_mean, y_mean, z_mean])
                pd_center = pd_points.mean(axis=0)
                distance = np.sum((gt_center - pd_center) ** 2)
                if distance < 1:
                    all_scene_correspondences += [(t, q, distance, box_vertices_non_aligned_fixed[q])]

        all_pred_points = []
        all_labels = []
        for k, correspondence in enumerate(all_scene_correspondences):
            shape_ply = shape_plys[correspondence[0]]
            shape_label = all_partnet_labels[correspondence[0]]
            instance_id = instance_ids[correspondence[0]]
            partnet_id = partnet_ids[correspondence[0]]

            instance_mask_pred = -np.ones(len(scan_points))

            pred_points_world = box_vertices_non_aligned_fixed[correspondence[1]]
            tree = cKDTree(pred_points_world)
            min_dist, min_idx = tree.query(scan_points)

            for is_near, i_nn, i_point in zip(min_dist <= 0.40, min_idx, range(len(scan_points))):
                if is_near:
                    instance_mask_pred[i_point] = object_id

            pred_points = scan_points[np.where(instance_mask_pred == object_id)[0]]

            shape_transform = shape_transforms[correspondence[0]]
            partnet_transform = partnet_transforms[correspondence[0]]

            # WORLD -> SHAPENET -> PARTNET
            pred_points = apply_inverse_transform(pred_points, shape_transform, partnet_transform)

            tree = cKDTree(shape_ply)
            min_dist, min_idx = tree.query(pred_points)
            min_idx_over = min_dist > 0.07
            result_labels = shape_label[min_idx]
            result_labels[min_idx_over] = -1

            # MAP semantic labels
            try:
                result_semantic_labels = np.array([
                    global_id_to_semantic_class[g_id] for g_id in shape_label[min_idx]
                ])
            except KeyError:
                print('Semantic labels not found')
                continue

            # MAP instance labels
            from_semantic_label_to_instance = dict(
                zip(sorted(list(set(result_semantic_labels))),
                range(len(set(result_semantic_labels))))
            )
            result_instance_labels = np.array([
                from_semantic_label_to_instance[s_id] for s_id in result_semantic_labels
            ])

            result_semantic_labels[min_idx_over] = -1
            result_instance_labels[min_idx_over] = -1

            all_pred_points += [pred_points]
            all_labels += [result_semantic_labels]

            if len(pred_points) == 0:
                continue

            min_1 = deepcopy(pred_points.min(0))
            pred_points -= pred_points.min(0)
            max_1 = deepcopy(pred_points.max() / 0.95)
            max_1 = np.array(max_1)
            pred_points /= pred_points.max() / 0.95
            max_2 = deepcopy(pred_points.max(0) / 2)
            pred_points -= pred_points.max(0) / 2

            mag_factor = mag_factors[args.category]
            pred_points_enlarged = pred_points * mag_factor

            voxel_transform = voxel_transforms[correspondence[0]]
            pred_grid = apply_inverse_transform(pred_points_enlarged, voxel_transform)
            pred_grid_int = pred_grid.astype('int')
            pred_grid_int = np.maximum(0, np.minimum(size - 1, pred_grid_int))

            voxel_points = np.zeros((size, size, size)).astype('uint8')
            voxel_points[pred_grid_int[:, 2], pred_grid_int[:, 1], pred_grid_int[:, 0]] = 1

            min_1_scannet = np.load(os.path.join(args.scannet_processed_dir, f'{partnet_id}_{scan_id}_{instance_id}_min_1.npy'))
            max_1_scannet = np.load(os.path.join(args.scannet_processed_dir, f'{partnet_id}_{scan_id}_{instance_id}_max_1.npy'))
            max_2_scannet = np.load(os.path.join(args.scannet_processed_dir, f'{partnet_id}_{scan_id}_{instance_id}_max_2.npy'))
            partnet_id = partnet_ids[correspondence[0]]
            gt_mask = np.load(os.path.join(args.partnet_processed_trees, f'{args.category}_geo', f'{partnet_id}_full.npy'))

            gt_mask_corrected = perform_correction(gt_mask[0], voxel_transform,
                                                   min_1_scannet, max_1_scannet, max_2_scannet,
                                                   min_1, max_1, max_2, mag_factor, args.voxel_dim)
            all_buf += [(gt_mask_corrected, voxel_points)]

            metric = iou(gt_mask[0], gt_mask_corrected)
            if metric >= 0.18:

                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{instance_id}_min_1.npy'), min_1)
                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{instance_id}_max_1.npy'), max_1)
                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{instance_id}_max_2.npy'), max_2)
                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{instance_id}_iou.npy'), metric)

                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{instance_id}.npy'), voxel_points)
                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{instance_id}_scan.npy'), pred_grid)
                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{instance_id}_sem_labels.npy'), result_semantic_labels)
                np.save(os.path.join(save_dir, f'{partnet_id}_{scan_id}_{instance_id}_ins_labels.npy'), result_instance_labels)

                split_ids += [f'{partnet_id}_{scan_id}_{instance_id}']

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
    parser.add_argument('--scannet_processed_dir', required=True, help='path to voxel ScanNet crops processed with prepare_scannet_samples.py')
    parser.add_argument('--partnet_processed_trees', required=True, help='path to voxelized PartNet trees processed with prepare_trees.py')
    parser.add_argument('--mlcvnet_output', required=True, help='path to MLCVNet .ply bboxes')
    parser.add_argument('--mode', default='train', help='ScanNet split to process')

    args = parser.parse_args()

    SAVE_DIR = os.path.join(args.save_dir, f'{args.category}_scannet_geo_finaltable')
    os.makedirs(SAVE_DIR, exist_ok=True)

    scan2cad_anno, \
    global_labels, \
    partnet_to_shapenet_transforms, \
    scannet_train_scenes, \
    scannet_val_scenes, \
    valid_partnet_ids, \
    mlcvnet_bboxes_filenames, \
    global_id_to_semantic_class = load_metadata(args)

    process_mlcvnet(args, scannet_train_scenes, scannet_val_scenes,
                    scan2cad_anno, global_labels, partnet_to_shapenet_transforms, valid_partnet_ids,
                    global_id_to_semantic_class, SAVE_DIR)