import argparse
import os, sys
import pickle
import json
from json import JSONDecodeError

import pandas as pd
import numpy as np
from tqdm import tqdm

import main_directories as md
import tree_processing as tp

sys.path.append('..')
from datasets.shapenet import VoxelisedShapeNetGNNDataset

shapenet_categories = ['04379243',
                       '02808440',
                       '02818832',
                       '02747177',
                       '02773838',
                       '02871439',
                       '02933112',
                       '04256520',
                       '03337140',
                       '03001627',
                       '02801938',
                       '03642806',
                       '03211117']

replace_dict = {'back_soft_surface': 'back_surface',
                'back_hard_surface': 'back_surface',
                'seat_soft_surface': 'seat_surface',
                'seat_hard_surface': 'seat_surface',
                'arm_sofa_style': 'arm_surface',
                'armrest_hard_surface': 'arm_surface',
                'back_holistic_frame': 'back_frame',
                'runner': 'bar_stretcher',
                'foot_base': 'regular_leg_base',
                'short_leg': 'leg',
                'back_connector': 'back_support',
                'pedestal_base': 'surface_base',
                'ground_surface': 'side_surface',
                'screen_frame': 'frame',
                'glass': 'tabletop_surface',
                'frame_horizontal_hard_surface': 'bed_frame_horizontal_surface',
                'bed_side_surface_panel': 'bed_side_surface',
                'other_leaf': 'other'}

parts_to_remove = ['caster_stem',
                   'wheel',
                   'rocker',
                   'runner',
                   'knob',
                   'lever',
                   'mechanical_control',
                   'back_surface_vertical_bar',
                   'back_surface_horizontal_bar',
                   'ground_surface',
                   'side_surface',
                   'bar',
                   'star_leg_set',
                   'runner',
                   'drawer',
                   'footrest']

parts_to_add = ['surface_base',
                'caster',
                'arm_horizontal_bar',
                'arm_vertical_bar',
                'vertical_side_panel',
                'central_support',
                'bottom_panel']

parts_to_squeeze = ['regular_table',
                    'table_base',
                    'tabletop',
                    'bed_unit',
                    'bed_frame',
                    'display_screen',
                    'screen_side']

voxel_name = 'full_vox_filled.colored.pkl'


def create_mask_map(mask, shape_labels):

    unique_mask_values = np.unique(mask)
    unique_mask_values = unique_mask_values[unique_mask_values > 0]
    global_id_to_name = {}
    for global_id in unique_mask_values:
        global_id_to_name[global_id] = \
            list(shape_labels[shape_labels['global_id'] == global_id]['part_dir_name'])[0]
    name_to_global_id = {global_id_to_name[k]: k for k in global_id_to_name}

    return name_to_global_id


def load_metadata():

    hier_dirs = sorted([x for x in os.listdir(md.STRUCT_DATA_DIR) if x.endswith('_hier')])
    geo_dirs = sorted([x for x in os.listdir(md.STRUCT_DATA_DIR) if x.endswith('_geo')])
    struct_categories = [x.split('_')[0] for x in hier_dirs]
    hier_to_localid = {struct_categories[i] for i in range(len(struct_categories))}

    struct_partnet_ids = []
    for hier_dir in hier_dirs:
        partnet_ids = [x.split('.')[0] for x in os.listdir(os.path.join(md.STRUCT_DATA_DIR, hier_dir)) \
                       if x.endswith('.json')]
        struct_partnet_ids += sorted([partnet_ids])

    with open(os.path.join(md.DICTIONARIES, 'parts_to_shapes.json'), 'rb') as fin:
        shapes_to_parts = json.load(fin)

    shapenet_categories = [x for x in os.listdir(md.SHAPENET_FULL) if len(x) == 8]
    shapenet_objects = {x: os.listdir(os.path.join(md.SHAPENET_FULL, x)) for x in shapenet_categories}
    obj_to_cat = {}
    for cat_id in shapenet_objects:
        for obj_id in shapenet_objects[cat_id]:
            obj_to_cat[obj_id] = cat_id

    voxel_partnet_ids = os.listdir(md.PARTNET_VOXELIZED)
    needed_partnet_ids = []
    for partnet_ids in struct_partnet_ids:
        needed_partnet_ids += [[x for x in sorted(list(set(partnet_ids)))
                                if x in shapes_to_parts and x in voxel_partnet_ids]]
    global_labels = pd.read_csv(os.path.join(md.DICTIONARIES, 'FULL_part_id_to_parts_description_11.csv'))

    return needed_partnet_ids, shapes_to_parts, obj_to_cat, global_labels, hier_to_localid


def analyse_part_data(localid, needed_partnet_ids, shapes_to_parts, obj_to_cat, global_labels):

    num_shapes_with_part_per_cat = []
    voxel_part_ratios_per_cat = []

    cat_ids = {}
    used_partnet_ids = []
    partnet_ids = needed_partnet_ids[localid]

    num_shapes_with_part = {}
    voxel_part_ratios = {}
    for j, partnet_id in enumerate(tqdm(partnet_ids)):
        try:
            with open(os.path.join(md.PARTNET, partnet_id, 'result.json'), 'rb') as fin:
                partnet_hier = json.load(fin)[0]
            obj_id = shapes_to_parts[partnet_id]
            cat_id = obj_to_cat[obj_id]
            if cat_id not in cat_ids:
                cat_ids[cat_id] = [partnet_id]
            else:
                cat_ids[cat_id] += [partnet_id]
            used_partnet_ids += [partnet_id]
            voxel_path = os.path.join(md.PARTNET_VOXELIZED, partnet_id, voxel_name)
            shape_pkl = pickle.load(open(voxel_path, 'rb'))
            mask = shape_pkl['mask']
            shape_labels = global_labels[global_labels['object_id'] == obj_id]

            try:
                name_to_global_id = create_mask_map(mask, shape_labels)
            except IndexError:
                print('IndexError in PartNet shape ', partnet_id)
                continue

            tp.propagate_leaves(partnet_hier, name_to_global_id)
            num_nodes = tp.add_dfs_id(partnet_hier)
            masks = tp.prepare_masks(partnet_hier, mask, num_nodes)

            dangling_ids = [i for i, x in enumerate(masks) if x.sum() == 0]
            if 0 not in dangling_ids:
                num_nodes = tp.add_dfs_id(partnet_hier)
                masks = tp.prepare_masks(partnet_hier, mask, num_nodes)

                all_node_names = []
                all_node_full_names = {}
                all_node_names, all_node_full_names = tp.collect_names(partnet_hier, all_node_names,
                                                                       all_node_full_names,
                                                                       leaf_name='name', full_name='')
                all_node_full_names = {key: partnet_hier['name'] + all_node_full_names[key] for key in
                                       all_node_full_names}
                all_node_full_names[0] = partnet_hier['name']
                unique_full_names = list(set(all_node_full_names.values()))
                for name in unique_full_names:
                    if name in num_shapes_with_part:
                        num_shapes_with_part[name] += 1
                    else:
                        num_shapes_with_part[name] = 1

                for key in all_node_full_names:
                    if all_node_full_names[key] in voxel_part_ratios:
                        voxel_part_ratios[all_node_full_names[key]] += masks[key].sum() / masks[0].sum()
                    else:
                        voxel_part_ratios[all_node_full_names[key]] = masks[key].sum() / masks[0].sum()
        except JSONDecodeError:
            continue

    num_shapes_with_part_per_cat += [num_shapes_with_part]
    voxel_part_ratios_per_cat += [voxel_part_ratios]

    return voxel_part_ratios


def select_parts(voxel_part_ratios):

    # calculate voxel ratios for parts
    voxel_part_ratios_thr = {key: voxel_part_ratios[key] for key in voxel_part_ratios if voxel_part_ratios[key] > 0}
    valid_parts = [x.split('/')[-1] for x in list(voxel_part_ratios_thr.keys())]
    valid_parts += ['other']

    # print all part ratios
    for part_name, ratio in voxel_part_ratios_thr.items():
        print(part_name, ': ', ratio)

    valid_parts = list(set(valid_parts) - set(parts_to_remove))
    valid_parts = list(set(valid_parts + parts_to_add))

    return valid_parts


def filter_trees(localid, valid_parts, needed_partnet_ids, shapes_to_parts, obj_to_cat, global_labels, args):

    old_hiers = []
    new_hiers = []

    partnet_ids = needed_partnet_ids[localid]

    max_children = {}
    max_masks = {}
    partnet_ids_valid = []
    SAVE_HIER_CAT_DIR = os.path.join(args.save_dir, args.category + '_hier')
    SAVE_GEO_CAT_DIR = os.path.join(args.save_dir, args.category + '_geo')
    os.makedirs(SAVE_HIER_CAT_DIR, exist_ok=True)
    os.makedirs(SAVE_GEO_CAT_DIR, exist_ok=True)

    for j, partnet_id in enumerate(tqdm(partnet_ids)):
        try:
            with open(os.path.join(md.PARTNET, partnet_id, 'result.json'), 'rb') as fin:
                partnet_hier = json.load(fin)[0]
            obj_id = shapes_to_parts[partnet_id]
            voxel_path = os.path.join(md.PARTNET_VOXELIZED, partnet_id, voxel_name)
            shape_pkl = pickle.load(open(voxel_path, 'rb'))
            mask = shape_pkl['mask']
            shape_labels = global_labels[global_labels['object_id'] == obj_id]

            try:
                name_to_global_id = create_mask_map(mask, shape_labels)
            except IndexError:
                print('IndexError in PartNet shape ', partnet_id)
                continue

            tp.propagate_leaves(partnet_hier, name_to_global_id)

            tp.squeeze_tree(partnet_hier)
            num_nodes = tp.add_dfs_id(partnet_hier)
            masks = tp.prepare_masks(partnet_hier, mask, num_nodes)

            dangling_ids = [i for i, x in enumerate(masks) if x.sum() == 0]
            if 0 not in dangling_ids:
                tp.delete_nodes(partnet_hier, dangling_ids)
                tp.replace_nodes(partnet_hier, replace_dict)
                tp.cut_tree(partnet_hier, valid_parts)
                tp.replace_nodes(partnet_hier, replace_dict)
                tp.add_extra_children(partnet_hier)

                old_hiers += [partnet_hier]
                tp.squeeze_the_only_leaves(partnet_hier)
                num_nodes = tp.add_dfs_id(partnet_hier)
                masks = tp.prepare_masks(partnet_hier, mask, num_nodes)

                dangling_ids = [i for i, x in enumerate(masks) if x.sum() == 0]
                tp.delete_nodes(partnet_hier, dangling_ids)
                tp.squeeze_the_only_leaves(partnet_hier)
                num_nodes = tp.add_dfs_id(partnet_hier)
                new_hiers += [partnet_hier]

                masks = tp.prepare_masks(partnet_hier, mask, num_nodes)
                dangling_ids = [i for i, x in enumerate(masks) if x.sum() == 0]
                assert len(dangling_ids) == 0
                masks = tp.prepare_masks(partnet_hier, mask, num_nodes)
                assert len(masks) == num_nodes

                tp.add_dfs_id(partnet_hier)
                tp.squeeze_parts(partnet_hier, parts_to_squeeze)

                tp.keep_first_children(partnet_hier)
                tp.merge_instances_full(partnet_hier)
                num_nodes = tp.add_dfs_id(partnet_hier)
                masks = tp.prepare_masks(partnet_hier, mask, num_nodes)

                if args.category == 'chair':
                    tp.unmerge_chair_arms(partnet_hier, masks, size=args.voxel_dim)

                tp.detect_relationships(partnet_hier, masks)
                max_masks[partnet_id] = len(masks)

                with open(os.path.join(os.path.join(SAVE_HIER_CAT_DIR, f'{partnet_id}.json')), 'w') as fout:
                    json.dump(partnet_hier, fout)
                np.save(os.path.join(SAVE_GEO_CAT_DIR, f'{partnet_id}.npy'), masks)
                np.save(os.path.join(SAVE_GEO_CAT_DIR, f'{partnet_id}_full.npy'), masks[0])

                partnet_ids_valid += [partnet_id]

                leaves = []
                tp.leaf_ids(partnet_hier, leaves)
                num_children = []
                tp.count_max_child(partnet_hier, num_children)
                if len(num_children) == 0:
                    max_children[partnet_id] = 0
                else:
                    max_children[partnet_id] = max(num_children)
        except JSONDecodeError:
            continue

    num_train = int(len(partnet_ids_valid) * 0.8)
    train_ids = np.random.choice(partnet_ids_valid, num_train, replace=False)
    val_ids = list(set(partnet_ids_valid) - set(train_ids))
    with open(os.path.join(SAVE_HIER_CAT_DIR, 'train.txt'), 'w') as f:
        for item in train_ids:
            f.write("%s\n" % item)
    with open(os.path.join(SAVE_HIER_CAT_DIR, 'val.txt'), 'w') as f:
        for item in val_ids:
            f.write("%s\n" % item)
    with open(os.path.join(SAVE_HIER_CAT_DIR, 'full.txt'), 'w') as f:
        for item in partnet_ids:
            f.write("%s\n" % item)

    return partnet_ids_valid


def traverse_trees(args):

    shapes_to_scans_path = os.path.join(md.DICTIONARIES, 'Shapenet_objects_in_Scannet_scans(Scan2CAD).json')
    parts_to_shapes_path = os.path.join(md.DICTIONARIES, 'PartNet_dir_of_ShapeNet_objects.json')
    cats_to_shapes_path = os.path.join(md.DICTIONARIES, 'Shapenet_cats_to_shapes.json')

    all_hier_strings = []

    data_features = ['object']
    hier_path = os.path.join(args.save_dir, args.category + '_hier')
    geo_path = os.path.join(args.save_dir, args.category + '_geo')
    dataset = 'full.txt'
    dataset = VoxelisedShapeNetGNNDataset(hier_path, geo_path, dataset, data_features, False,
                                          shapes_to_scans_path, os.path.join(md.DICTIONARIES, 'parts_to_shapes.json'),
                                          cats_to_shapes_path, md.PARTNET_VOXELIZED)

    hier_full_ids = []

    all_occurences = {}
    for j in tqdm(range(len(dataset))):
        traversal_nodes = dataset[j][2][0].root.depth_first_traversal()
        for node in traversal_nodes:
            hier_full_ids += [(node.full_label, node.is_leaf)]
            if node.full_label not in all_occurences:
                all_occurences[node.full_label] = 1
            else:
                all_occurences[node.full_label] += 1

    hier_full_ids = list(set(hier_full_ids))
    hier_names = []
    hier_names_to_leaf = {}
    for full_id in hier_full_ids:
        hier_names += [full_id[0]]
        if (full_id[1]):
            hier_names_to_leaf[full_id[0]] = 'leaf'
        else:
            hier_names_to_leaf[full_id[0]] = 'subcomponents'
    hier_names = list(set(hier_names))
    hier_names = sorted(hier_names)

    hier_strings = []
    for j in range(len(hier_names)):
        hier_strings += ['{} {} {}'.format(j + 1, hier_names[j], hier_names_to_leaf[hier_names[j]])]
    all_hier_strings += [hier_strings]

    with open(os.path.join(args.save_dir, args.category + '_hier', 'full_traverse.txt'), 'w') as f:
        for item in hier_strings:
            f.write("%s\n" % item)


if __name__ == '__main__':

    # params
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('--save_dir', required=True, help='path to store data')
    parser.add_argument('--category', required=True, help='shape category to process')
    parser.add_argument('--voxel_dim', type=int, default=32, help='size of voxelized shapes')

    args = parser.parse_args()

    needed_partnet_ids, shapes_to_parts, obj_to_cat, global_labels, hier_to_localid = load_metadata()
    localid = hier_to_localid[args.category]
    voxel_part_ratios = analyse_part_data(localid,
                                          needed_partnet_ids,
                                          shapes_to_parts,
                                          obj_to_cat,
                                          global_labels)

    valid_parts = select_parts(voxel_part_ratios)
    processed_partnet_ids = filter_trees(localid,
                                         valid_parts,
                                         needed_partnet_ids,
                                         shapes_to_parts,
                                         obj_to_cat,
                                         global_labels,
                                         args)


