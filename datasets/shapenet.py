from glob import glob
import pickle
import os
import json
import random
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# sys.path.append('/sources/Scan2CAD/Network/base/')
from datasets.sample_loader import load_sample
import datasets.transforms as transforms
from unet3d.hierarchy import Tree


class VoxelisedScanNetGNNDataset(Dataset):

    def __init__(self, root, geos, partnet_geos, object_list, data_features, load_geo=False,
                 shapes_to_scans_path=None, parts_to_shapes_path=None, cats_to_shapes_path=None,
                 shapenet_voxelized_path=None,
                 predict_part_centers=False):
        self.root = root
        self.geos = geos
        self.partnet_geos = partnet_geos
        self.data_features = data_features
        self.load_geo = load_geo
        self.predict_part_centers = predict_part_centers

        self.shapenet_voxelized_path = shapenet_voxelized_path

        if isinstance(object_list, str):
            with open(os.path.join(self.geos, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

        # load metadata
        with open(shapes_to_scans_path, 'rb') as fin:
            self.shapes_to_scans = json.load(fin)
        with open(parts_to_shapes_path, 'rb') as fin:
            self.parts_to_shapes = json.load(fin)
        self.shapes_to_parts = {self.parts_to_shapes[k]: k for k in self.parts_to_shapes}
        with open(cats_to_shapes_path, 'rb') as fin:
            self.cats_to_shapes = json.load(fin)
        self.shapes_to_cats = {}
        for cat_id in self.cats_to_shapes:
            for obj_id in self.cats_to_shapes[cat_id]:
                self.shapes_to_cats[obj_id] = cat_id

    def __getitem__(self, index):
        partnet_scannet_id = self.object_names[index]
        tokens = partnet_scannet_id.split('_')
        partnet_id = tokens[0]
        if 'object' in self.data_features:
            obj = self.load_object(os.path.join(self.root, partnet_id + '.json'),
                                   load_geo=self.load_geo)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (partnet_id,)
            else:
                assert False, 'ERROR: unknown feat type %s!' % feat

        voxel_path = os.path.join(self.shapenet_voxelized_path, partnet_id, 'full_vox.colored.pkl')
        shape_sdf = torch.FloatTensor(pickle.load(open(voxel_path, 'rb'))['sdf'])
        shape_mask = torch.FloatTensor(np.load(os.path.join(self.partnet_geos, f'{partnet_id}_full.npy')))
        scannet_geo = torch.FloatTensor(np.load(os.path.join(self.geos, f'{partnet_scannet_id}.npy')))

        output = (scannet_geo, shape_sdf, shape_mask, data_feats, partnet_id)

        return output

    def __len__(self):
        return len(self.object_names)

    def get_anno_id(self, anno_id):
        obj = self.load_object(os.path.join(self.root, anno_id + '.json'),
                               load_geo=self.load_geo)
        return obj

    @staticmethod
    def load_object(fn, load_geo=False):
        if load_geo:
            geo_fn = fn.replace('_hier', '_geo').replace('json', 'npy')
            geo_data = np.load(geo_fn)

        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            if 'complete_dfs_id' in node_json:
                node = Tree.Node(
                    part_id=node_json['dfs_id'],
                    complete_part_id=node_json['complete_dfs_id'],
                    is_leaf=('children' not in node_json),
                    label=node_json['name'])
            else:
                node = Tree.Node(
                    part_id=node_json['dfs_id'],
                    is_leaf=('children' not in node_json),
                    label=node_json['name'])

            if 'geo' in node_json.keys():
                node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32)

            if load_geo:
                node.geo = torch.tensor(geo_data[node_json['dfs_id']], dtype=torch.float32)

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if 'edges' in node_json:
                for edge in node_json['edges']:
                    if 'params' in edge:
                        edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
                    node.edges.append(edge)

            if parent is None:
                root = node
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx + 1 - len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)

        return obj


def generate_scannet_datasets(hierarchies=None, geos=None, partnet_geos=None,
                              train_samples='train.txt', val_samples='val.txt',
                              data_features=('object',), load_geo=True,
                              shapes_to_scans_path=None, parts_to_shapes_path=None,
                              cats_to_shapes_path=None, shapenet_voxelized_path=None,
                              predict_part_centers=False,
                              **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedScanNetGNNDataset

    train_dataset = Dataset(hierarchies, geos, partnet_geos, train_samples, data_features, load_geo,
                            shapes_to_scans_path=shapes_to_scans_path,
                            parts_to_shapes_path=parts_to_shapes_path,
                            cats_to_shapes_path=cats_to_shapes_path,
                            shapenet_voxelized_path=shapenet_voxelized_path,
                            predict_part_centers=predict_part_centers)
    val_dataset = Dataset(hierarchies, geos, partnet_geos, val_samples, data_features, load_geo,
                          shapes_to_scans_path=shapes_to_scans_path,
                          parts_to_shapes_path=parts_to_shapes_path,
                          cats_to_shapes_path=cats_to_shapes_path,
                          shapenet_voxelized_path=shapenet_voxelized_path,
                          predict_part_centers=predict_part_centers)

    return {
        'train': train_dataset,
        'val': val_dataset
    }


class VoxelisedScanNetAllShapesGNNDataset(Dataset):

    def __init__(self, root, partnet_to_dirs_path, object_list, data_features, load_geo=False,
                 shapes_to_scans_path=None, parts_to_shapes_path=None, cats_to_shapes_path=None,
                 shapenet_voxelized_path=None):
        self.root = root
        self.data_features = data_features
        self.load_geo = load_geo
        self.shapenet_voxelized_path = shapenet_voxelized_path

        with open(partnet_to_dirs_path, 'rb') as f:
            self.partnet_to_dirs = pickle.load(f)

        if isinstance(object_list, str):
            with open(os.path.join(self.root, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

        # load metadata
        with open(shapes_to_scans_path, 'rb') as fin:
            self.shapes_to_scans = json.load(fin)
        with open(parts_to_shapes_path, 'rb') as fin:
            self.parts_to_shapes = json.load(fin)
        self.shapes_to_parts = {self.parts_to_shapes[k]: k for k in self.parts_to_shapes}
        with open(cats_to_shapes_path, 'rb') as fin:
            self.cats_to_shapes = json.load(fin)
        self.shapes_to_cats = {}
        for cat_id in self.cats_to_shapes:
            for obj_id in self.cats_to_shapes[cat_id]:
                self.shapes_to_cats[obj_id] = cat_id

    def __getitem__(self, index):
        partnet_scannet_id = self.object_names[index]
        tokens = partnet_scannet_id.split('_')
        partnet_id = tokens[0]
        common_path = self.partnet_to_dirs[partnet_id]
        id_without_rotation = '_'.join(tokens[:-1])
        if 'object' in self.data_features:
            geo_fn = os.path.join(common_path + '_geo', f'{partnet_id}.npy')
            obj = self.load_object(os.path.join(common_path+'_hier', partnet_id + '.json'),
                                   load_geo=self.load_geo, geo_fn=geo_fn)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (partnet_id,)
            else:
                assert False, 'ERROR: unknown feat type %s!' % feat

        voxel_path = os.path.join(self.shapenet_voxelized_path, partnet_id, 'full_vox.colored.pkl')
        shape_sdf = torch.FloatTensor(pickle.load(open(voxel_path, 'rb'))['sdf'])

        suffix_mask = '_geo'
        suffix_scannet_geo = '_scannet_geo_mlcvnet_finaltable_1'
        shape_mask = torch.FloatTensor(np.load(os.path.join(common_path+suffix_mask, f'{partnet_id}_full.npy')))
        scannet_geo = torch.FloatTensor(np.load(os.path.join(common_path+suffix_scannet_geo, f'{partnet_scannet_id}.npy')))

        output = (scannet_geo, shape_sdf, shape_mask, data_feats, partnet_id, 0, tokens)

        return output

    def __len__(self):
        return len(self.object_names)

    def get_anno_id(self, anno_id):
        obj = self.load_object(os.path.join(self.root, anno_id + '.json'),
                               load_geo=self.load_geo)
        return obj

    @staticmethod
    def load_object(fn, load_geo=False, geo_fn=None):
        if load_geo:
            geo_data = np.load(geo_fn)

        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            if 'complete_dfs_id' in node_json:
                node = Tree.Node(
                    part_id=node_json['dfs_id'],
                    complete_part_id=node_json['complete_dfs_id'],
                    is_leaf=('children' not in node_json),
                    label=node_json['name'])
            else:
                node = Tree.Node(
                    part_id=node_json['dfs_id'],
                    is_leaf=('children' not in node_json),
                    label=node_json['name'])

            if 'geo' in node_json.keys():
                node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32)

            if load_geo:
                node.geo = torch.tensor(geo_data[node_json['dfs_id']], dtype=torch.float32)

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if 'edges' in node_json:
                for edge in node_json['edges']:
                    if 'params' in edge:
                        edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
                    node.edges.append(edge)

            if parent is None:
                root = node
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx + 1 - len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)

        return obj


def generate_scannet_allshapes_datasets(root=None, partnet_to_dirs_path=None,
                              train_samples='train.txt', val_samples='val.txt',
                              data_features=('object',), load_geo=True,
                              shapes_to_scans_path=None, parts_to_shapes_path=None,
                              cats_to_shapes_path=None, shapenet_voxelized_path=None,
                              **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedScanNetAllShapesGNNDataset

    train_dataset = Dataset(root, partnet_to_dirs_path, train_samples, data_features, load_geo,
                            shapes_to_scans_path=shapes_to_scans_path,
                            parts_to_shapes_path=parts_to_shapes_path,
                            cats_to_shapes_path=cats_to_shapes_path,
                            shapenet_voxelized_path=shapenet_voxelized_path)
    val_dataset = Dataset(root, partnet_to_dirs_path, val_samples, data_features, load_geo,
                          shapes_to_scans_path=shapes_to_scans_path,
                          parts_to_shapes_path=parts_to_shapes_path,
                          cats_to_shapes_path=cats_to_shapes_path,
                          shapenet_voxelized_path=shapenet_voxelized_path)

    return {
        'train': train_dataset,
        'val': val_dataset
    }


class VoxelisedScanNetAllShapesRotGNNDataset(Dataset):

    def __init__(self, root, partnet_to_dirs_path, object_list, data_features, load_geo=False,
                 shapes_to_scans_path=None, parts_to_shapes_path=None, cats_to_shapes_path=None,
                 shapenet_voxelized_path=None):
        self.root = root
        self.data_features = data_features
        self.load_geo = load_geo
        self.shapenet_voxelized_path = shapenet_voxelized_path

        with open(partnet_to_dirs_path, 'rb') as f:
            self.partnet_to_dirs = pickle.load(f)

        if isinstance(object_list, str):
            with open(os.path.join(self.root, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

        # load metadata
        with open(shapes_to_scans_path, 'rb') as fin:
            self.shapes_to_scans = json.load(fin)
        with open(parts_to_shapes_path, 'rb') as fin:
            self.parts_to_shapes = json.load(fin)
        self.shapes_to_parts = {self.parts_to_shapes[k]: k for k in self.parts_to_shapes}
        with open(cats_to_shapes_path, 'rb') as fin:
            self.cats_to_shapes = json.load(fin)
        self.shapes_to_cats = {}
        for cat_id in self.cats_to_shapes:
            for obj_id in self.cats_to_shapes[cat_id]:
                self.shapes_to_cats[obj_id] = cat_id

    def __getitem__(self, index):
        partnet_scannet_id = self.object_names[index]
        tokens = partnet_scannet_id.split('_')
        partnet_id = tokens[0]
        rotation = tokens[4]
        id_without_rotation = '_'.join(tokens[:-1])
        common_path = self.partnet_to_dirs[partnet_id]
        if 'object' in self.data_features:
            geo_fn = os.path.join(common_path+'_geo_8rot', partnet_id + f'_{rotation}.npy')
            obj = self.load_object(os.path.join(common_path+'_hier', partnet_id + '.json'),
                                   load_geo=self.load_geo, rotation=rotation,
                                   geo_fn=geo_fn)

        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (partnet_id,)
            else:
                assert False, 'ERROR: unknown feat type %s!' % feat

        voxel_path = os.path.join(self.shapenet_voxelized_path, partnet_id, 'full_vox.colored.pkl')
        shape_sdf = torch.FloatTensor(pickle.load(open(voxel_path, 'rb'))['sdf'])

        suffix_mask = '_geo_8rot'
        suffix_scannet_geo = '_scannet_geo_mlcvnet_8rot'
        shape_mask = torch.FloatTensor(np.load(os.path.join(common_path+suffix_mask, f'{partnet_id}_full_{rotation}.npy')))
        scannet_geo = torch.FloatTensor(np.load(os.path.join(common_path+suffix_scannet_geo, f'{partnet_scannet_id}.npy')))

        output = (scannet_geo, shape_sdf, shape_mask, data_feats, partnet_id, int(rotation), tokens)

        return output

    def __len__(self):
        return len(self.object_names)

    def get_anno_id(self, anno_id):
        obj = self.load_object(os.path.join(self.root, anno_id + '.json'),
                               load_geo=self.load_geo)
        return obj

    @staticmethod
    def load_object(fn, load_geo=False, rotation=None, geo_fn=None):
        if load_geo:
            geo_data = np.load(geo_fn)

        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            if 'complete_dfs_id' in node_json:
                node = Tree.Node(
                    part_id=node_json['dfs_id'],
                    complete_part_id=node_json['complete_dfs_id'],
                    is_leaf=('children' not in node_json),
                    label=node_json['name'])
            else:
                node = Tree.Node(
                    part_id=node_json['dfs_id'],
                    is_leaf=('children' not in node_json),
                    label=node_json['name'])

            if 'geo' in node_json.keys():
                node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32)

            if load_geo:
                node.geo = torch.tensor(geo_data[node_json['dfs_id']], dtype=torch.float32)

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if 'edges' in node_json:
                for edge in node_json['edges']:
                    if 'params' in edge:
                        edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
                    node.edges.append(edge)

            if parent is None:
                root = node
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx + 1 - len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)

        return obj


def generate_scannet_allshapes_rot_datasets(root=None, partnet_to_dirs_path=None,
                              train_samples='train.txt', val_samples='val.txt',
                              data_features=('object',), load_geo=True,
                              shapes_to_scans_path=None, parts_to_shapes_path=None,
                              cats_to_shapes_path=None, shapenet_voxelized_path=None,
                              **kwargs):
    if isinstance(data_features, str):
        data_features = [data_features]

    Dataset = VoxelisedScanNetAllShapesRotGNNDataset

    train_dataset = Dataset(root, partnet_to_dirs_path, train_samples, data_features, load_geo,
                            shapes_to_scans_path=shapes_to_scans_path,
                            parts_to_shapes_path=parts_to_shapes_path,
                            cats_to_shapes_path=cats_to_shapes_path,
                            shapenet_voxelized_path=shapenet_voxelized_path)
    val_dataset = Dataset(root, partnet_to_dirs_path, val_samples, data_features, load_geo,
                          shapes_to_scans_path=shapes_to_scans_path,
                          parts_to_shapes_path=parts_to_shapes_path,
                          cats_to_shapes_path=cats_to_shapes_path,
                          shapenet_voxelized_path=shapenet_voxelized_path)

    return {
        'train': train_dataset,
        'val': val_dataset
    }
