import argparse
import os, sys
import json
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import nltk
from nltk.cluster import KMeansClusterer

from ..datasets.shapenet import VoxelisedScanNetAllShapesGNNDataset
from part_to_id import part_dict


def collect_all_masks(args):

    category = args.category
    if category == 'storage_furniture':
        category = 'storagefurniture'
    size = args.voxel_dim
    shapes_dict = part_dict[category]
    used_partnet_ids = []
    shapes = {i: [] for i in range(len(shapes_dict))}

    CAT_DIR = os.path.join(args.partnet_processed_trees, args.category)
    for filename in tqdm(os.listdir(CAT_DIR + '_hier')):
        partnet_id = filename.split('.')[0]
        if filename.endswith('.json'):
            geo_fn = os.path.join(CAT_DIR + '_geo', f'{partnet_id}.npy')
            gt_tree = VoxelisedScanNetAllShapesGNNDataset.load_object(os.path.join(CAT_DIR + '_hier', partnet_id + '.json'),
                                                                      load_geo=True, geo_fn=geo_fn)

            part = np.zeros((size, size, size))
            try:
                if partnet_id not in used_partnet_ids:
                    chair_arms = []
                    for child in gt_tree.root.children:
                        if child.label != 'chair_arm' and \
                           child.label != 'chair_arm_left' and \
                           child.label != 'chair_arm_right':
                            geometry = child.geo.numpy()[0]
                            shapes[shapes_dict[child.label]] += [geometry.reshape((-1))]
                        else:
                            geometry = child.geo.numpy()[0]
                            chair_arms += [geometry]
                            part += geometry
                    for chair_arm in chair_arms:
                        coords = np.stack(np.where(chair_arm)).T
                        if coords.mean(axis=0)[2] < size / 2:
                            shapes[0] += [chair_arm.reshape((-1))]
                        else:
                            shapes[1] += [chair_arm.reshape((-1))]
                    used_partnet_ids += [partnet_id]
            except KeyError:
                print('Exception raised for shape with PartNet id', partnet_id)
                continue
    return shapes


def run_kmeans(args, shapes):

    # compute priors with KMeans for each shape part
    category = args.category
    if category == 'storage_furniture':
        category = 'storagefurniture'
    size = args.voxel_dim
    shapes_dict = part_dict[category]

    kmeans_centroids = {i: [] for i in range(len(shapes_dict))}
    all_assigned_clusters = []
    for i in tqdm(shapes):
        kclusterer = KMeansClusterer(10, distance=nltk.cluster.util.euclidean_distance, repeats=30,
                                     avoid_empty_clusters=True)
        if len(shapes[i]) >= 10:
            assigned_clusters = kclusterer.cluster(shapes[i], assign_clusters=True)
            all_assigned_clusters += [assigned_clusters]
            cluster_centers = [x.reshape((size, size, size)) for x in kclusterer.means()]
            kmeans_centroids[i] = cluster_centers
        else:
            kmeans_centroids[i] = [x.reshape((size, size, size)) for x in shapes[i]]

    # normalize clusters
    kmeans_centroids_normed = {i: [] for i in range(len(shapes_dict))}
    for i in kmeans_centroids:
        for centroid in kmeans_centroids[i]:
            centroid_normed = (centroid - centroid.min()) / (centroid.max() - centroid.min())
            kmeans_centroids_normed[i] += [centroid_normed]

    # fill centroids with zeros if needed
    for i in kmeans_centroids_normed:
        if len(kmeans_centroids_normed[i]) < 10:
            for j in range(10 - len(kmeans_centroids_normed[i])):
                kmeans_centroids_normed[i] += [np.zeros((size, size, size))]

    # save priors
    os.makedirs(args.save_dir, exist_ok=True)
    for i in range(len(shapes_dict)):
        stacked_priors = np.stack(kmeans_centroids_normed[i])
        np.save(os.path.join(args.save_dir, f'{i}.npy'), stacked_priors)


if __name__ == '__main__':

    # params
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('--save_dir', required=True, help='path to store data')
    parser.add_argument('--category', required=True, help='shape category to process')
    parser.add_argument('--voxel_dim', type=int, default=32, help='size of voxelized shapes')
    parser.add_argument('--partnet_processed_trees', required=True, help='path to voxelized PartNet trees processed with prepare_trees.py')

    args = parser.parse_args()

    shapes = collect_all_masks(args)
    run_kmeans(args, shapes)