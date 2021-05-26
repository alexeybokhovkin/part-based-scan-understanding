import argparse
import os, sys
import pickle
from shutil import copyfile


def gather_shapes(args, data_suffix, categories_dict):

    all_trains = []
    all_vals = []
    all_parts = []
    all_is_leaf = []
    partnet_to_dirs = {}
    parts_to_ids = {}
    all_partnet_ids = []

    for category in categories_dict:
        cat_dir = categories_dict[category]
        cat_name = cat_dir.split('/')[-1]
        with open(os.path.join(cat_dir + data_suffix, 'train.txt'), 'r') as fin:
            lines = fin.readlines()
            lines = [x[:-1] for x in lines]
            all_trains += lines
            partnet_ids = [x.split('_')[0] for x in lines]
            partnet_to_dirs_local = {x: cat_dir for x in partnet_ids}
            partnet_to_dirs.update(partnet_to_dirs_local)
            all_partnet_ids += partnet_ids
        with open(os.path.join(cat_dir + data_suffix, 'val.txt'), 'r') as fin:
            lines = fin.readlines()
            lines = [x[:-1] for x in lines]
            all_vals += lines
            partnet_ids = [x.split('_')[0] for x in lines]
            partnet_to_dirs_local = {x: cat_dir for x in partnet_ids}
            partnet_to_dirs.update(partnet_to_dirs_local)
            all_partnet_ids += partnet_ids
        with open(os.path.join(cat_dir + '_hier', 'full_traverse.txt'), 'r') as fin:
            lines = fin.readlines()
            lines = [x[:-1] for x in lines]
            parts = [x.split(' ')[1] for x in lines]
            is_leaf = [x.split(' ')[2] for x in lines]
            all_parts += parts
            all_is_leaf += is_leaf

    with open(os.path.join(args.save_all_shapes_dir, 'train.txt'), 'w') as f:
        for item in all_trains:
            f.write("%s\n" % item)
    with open(os.path.join(args.save_all_shapes_dir, 'val.txt'), 'w') as f:
        for item in all_vals:
            f.write("%s\n" % item)
    with open(os.path.join(args.save_all_shapes_dir, 'full.txt'), 'w') as f:
        for item in all_trains + all_vals:
            f.write("%s\n" % item)

    with open(os.path.join(args.save_all_shapes_dir, 'partnet_to_dirs.pkl'), 'wb') as f:
        pickle.dump(partnet_to_dirs, f, pickle.HIGHEST_PROTOCOL)

    hier_strings = []
    for j in range(len(all_parts)):
        hier_strings += ['{} {} {}'.format(j + 1, all_parts[j], all_is_leaf[j])]
    with open(os.path.join(args.save_all_shapes_dir, 'full_traverse.txt'), 'w') as f:
        for item in hier_strings:
            f.write("%s\n" % item)

    i = 0
    for j, part_name in enumerate(all_parts):
        if '/' in part_name:
            parts_to_ids[all_parts[j]] = i
            i += 1
        else:
            parts_to_ids[all_parts[j]] = i
    with open(os.path.join(args.save_all_shapes_dir, 'parts_to_priors.pkl'), 'wb') as f:
        pickle.dump(parts_to_ids, f, pickle.HIGHEST_PROTOCOL)

    return parts_to_ids


def gather_priors(args, priors_folders, parts_to_ids):

    for prior_folder in priors_folders:
        shape_name = prior_folder.split('_')[0]
        if shape_name == 'storagefurniture':
            shape_name = 'storage_furniture'
        if shape_name == 'trashcan':
            shape_name = 'trash_can'
        id_offset = parts_to_ids[shape_name]
        priors_dir_name = os.path.join(args.priors_dir, prior_folder)
        for prior_name in sorted(os.listdir(priors_dir_name)):
            prior_id = int(prior_name.split('.')[0])
            copyfile(os.path.join(priors_dir_name, prior_name),
                     os.path.join(args.save_all_priors_dir, str(prior_id + id_offset) + '.npy'))


if __name__ == '__main__':

    # params
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('--save_all_shapes_dir', required=True, help='path to store scan data specs')
    parser.add_argument('--save_all_priors_dir', required=True, help='path to store gathered priors data')
    parser.add_argument('--data_dir', required=True, help='path to directory with processed trees and scan crops')
    parser.add_argument('--priors_dir', required=True, help='path to directory with processed priors (after compute_priors.py)')

    args = parser.parse_args()

    categories_dict = {}
    categories_dict['chair'] = os.path.join(args.data_dir, 'chair')
    categories_dict['table'] = os.path.join(args.data_dir, 'table')
    categories_dict['storagefurniture'] = os.path.join(args.data_dir, 'storagefurniture')
    categories_dict['bed'] = os.path.join(args.data_dir, 'bed')
    categories_dict['trashcan'] = os.path.join(args.data_dir, 'trashcan')

    priors_folders = ['chair_32_filled',
                      'bed_32_filled',
                      'storagefurniture_32_filled',
                      'table_32_filled',
                      'trashcan_32_filled']

    os.makedirs(args.save_all_shapes_dir, exist_ok=True)
    os.makedirs(args.save_all_priors_dir, exist_ok=True)

    data_suffix = '_scannet_geo'
    # data_suffix = '_scannet_geo_mlcvnet'

    parts_to_ids = gather_shapes(args, data_suffix, categories_dict)
    gather_priors(args, priors_folders, parts_to_ids)
