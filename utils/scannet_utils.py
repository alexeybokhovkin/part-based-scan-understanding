import os
import pickle

import pandas as pd
import trimesh
import numpy as np

from .vox import load_sample


def get_global_part_labels_description(dictionaries_dir='dictionaries', return_full=False):
    csv_name = 'part_id_to_parts_description.csv'
    if return_full:
        csv_name = 'FULL_' + csv_name

    path = os.path.join(dictionaries_dir, csv_name)
    global_labels = pd.read_csv(path, index_col=0, dtype=str)

    global_labels['part_id'] = global_labels.part_id.astype(int)
    global_labels['set_id'] = global_labels.set_id.astype(int)

    return global_labels


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data


def get_scannet(scan_id, scannet_dir='datasets/scannet', output_type='vox'):
    assert output_type in ['vox', 'mesh', 'array']

    if output_type == 'vox':
        scan_path = os.path.join(scannet_dir, scan_id, scan_id + '.vox')
        vox = load_sample(scan_path)
        return vox
    else:
        scan_path = os.path.join(scannet_dir, scan_id, scan_id + '_vh_clean_2.labels.ply')
        mesh = trimesh.load_mesh(scan_path)

        if output_type == 'mesh':
            return mesh
        elif output_type == 'array':
            return np.array(mesh.vertices)