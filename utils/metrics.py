import numpy as np
from scipy.optimize import linear_sum_assignment
import os, json
from scipy.spatial.distance import cdist
import trimesh


def compare_two_nodes(gt_node, pd_node):
    compare_log_local = {}
    compare_log_local['n_children_diff'] = abs(len(gt_node.children) - len(pd_node.children))
    gt_labels = []
    for child in gt_node.children:
        gt_labels += [child.get_semantic_id()]
    pd_labels = []
    for child in pd_node.children:
        pd_labels += [child.get_semantic_id()]
    compare_log_local['gt_semantic_sup'] = len(set(gt_labels) - set(pd_labels))
    compare_log_local['pd_semantic_sup'] = len(set(pd_labels) - set(gt_labels))
    return compare_log_local


def compare_by_dfs(gt_root, pd_root, cur_level=0, compare_log=None):
    if 'pd_not_leaves_trav_len' not in compare_log:
        compare_log['pd_not_leaves_trav_len'] = 0
    if 'pd_spare_leaves_trav_len' not in compare_log:
        compare_log['pd_spare_leaves_trav_len'] = 0
    if cur_level not in compare_log:
        compare_log[cur_level] = []
    log = compare_two_nodes(gt_root, pd_root)
    compare_log[cur_level] += [log]
    gt_labels = []
    if len(gt_root.children) > 0:
        for child in gt_root.children:
            gt_labels += [child.get_semantic_id()]
    pd_labels = []
    if len(pd_root.children) > 0:
        for child in pd_root.children:
            pd_labels += [child.get_semantic_id()]

    assignment = {}
    gt_nodes = gt_root.children
    gt_nodes_with_labels = {}
    for i, node in enumerate(gt_nodes):
        if node.get_semantic_id() not in gt_nodes_with_labels:
            gt_nodes_with_labels[node.get_semantic_id()] = [i]
        else:
            gt_nodes_with_labels[node.get_semantic_id()] += [i]
    pred_nodes = pd_root.children
    pred_nodes_with_labels = {}
    for i, node in enumerate(pred_nodes):
        if node.get_semantic_id() not in pred_nodes_with_labels:
            pred_nodes_with_labels[node.get_semantic_id()] = [i]
        else:
            pred_nodes_with_labels[node.get_semantic_id()] += [i]
    for gt_label in gt_nodes_with_labels:
        if gt_label not in pred_nodes_with_labels:
            for node in gt_nodes_with_labels[gt_label]:
                assignment[node] = -1
        else:
            if len(gt_nodes_with_labels[gt_label]) == 1 and len(pred_nodes_with_labels[gt_label]) == 1:
                assignment[gt_nodes_with_labels[gt_label][0]] = pred_nodes_with_labels[gt_label][0]
            else:
                dist_matrix = np.zeros((len(gt_nodes_with_labels[gt_label]), len(pred_nodes_with_labels[gt_label])))
                for i, gt_node in enumerate(gt_nodes_with_labels[gt_label]):
                    for j, pred_node in enumerate(pred_nodes_with_labels[gt_label]):
                        gt_traversal = gt_nodes[gt_node].depth_first_traversal()
                        pred_traversal = pred_nodes[pred_node].depth_first_traversal()
                        distance = abs(len(gt_traversal) - len(pred_traversal))
                        dist_matrix[i, j] = distance
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                for i in range(len(row_ind)):
                    assignment[gt_nodes_with_labels[gt_label][row_ind[i]]] = pred_nodes_with_labels[gt_label][
                        col_ind[i]]

    for gt_node in assignment:
        if assignment[gt_node] != -1:
            compare_by_dfs(gt_root.children[gt_node], pd_root.children[assignment[gt_node]], cur_level + 1, compare_log)
    if len(gt_labels) == 0 and len(pd_labels) > 0:
        compare_log['pd_not_leaves_trav_len'] += len(pd_root.depth_first_traversal()) - 1
    for pd_sem_id in pred_nodes_with_labels:
        if pd_sem_id not in gt_nodes_with_labels:
            for pd_node in pred_nodes_with_labels[pd_sem_id]:
                compare_log['pd_spare_leaves_trav_len'] += len(pred_nodes[pd_node].depth_first_traversal())


def aggregate_dfs_log(log):
    global_log = {}
    for level in log:
        if isinstance(level, int):
            level_log = {}
            level_log['n_children_diff'] = 0
            level_log['gt_semantic_sup'] = 0
            level_log['pd_semantic_sup'] = 0
            for local_log in log[level]:
                level_log['n_children_diff'] += local_log['n_children_diff']
                level_log['gt_semantic_sup'] += local_log['gt_semantic_sup']
                level_log['pd_semantic_sup'] += local_log['pd_semantic_sup']
            global_log[level] = level_log
    global_log['pd_not_leaves_trav_len'] = log['pd_not_leaves_trav_len']
    global_log['pd_spare_leaves_trav_len'] = log['pd_spare_leaves_trav_len']
    return global_log


def compare_trees(gt_tree, pd_tree):
    gt_depth = gt_tree.depth()
    pd_depth = pd_tree.depth()

    compare_log = {}
    compare_by_dfs(gt_tree.root, pd_tree.root, 0, compare_log)
    compare_log = aggregate_dfs_log(compare_log)
    compare_log['gt_depth'] = gt_depth
    compare_log['pd_depth'] = pd_depth
    compare_log['depth_diff'] = gt_depth - pd_depth

    return compare_log


def aggregate_statistics(path):
    total_log = {}
    total_log_general = {}

    total_log_general['total'] = {}
    total_log_general['total']['depth_diff_num'] = 0
    total_log_general['total']['depth_diff_total'] = 0
    total_log_general['total']['depth_diff_num_ratio'] = 0
    total_log_general['total']['pd_not_leaves_num'] = 0
    total_log_general['total']['pd_not_leaves_total'] = 0
    total_log_general['total']['pd_not_leaves_num_ratio'] = 0
    total_log_general['total']['pd_spare_leaves_num'] = 0
    total_log_general['total']['pd_spare_leaves_total'] = 0
    total_log_general['total']['pd_spare_leaves_num_ratio'] = 0
    total_log_general['total']['n_objects'] = 0

    for i in range(5):
        total_log[str(i)] = {}
        total_log[str(i)]['n_children_diff_num'] = 0
        total_log[str(i)]['n_children_diff_total'] = 0
        total_log[str(i)]['n_children_diff_num_ratio'] = 0
        total_log[str(i)]['gt_semantic_sup_num'] = 0
        total_log[str(i)]['gt_semantic_sup_total'] = 0
        total_log[str(i)]['gt_semantic_sup_num_ratio'] = 0
        total_log[str(i)]['pd_semantic_sup_num'] = 0
        total_log[str(i)]['pd_semantic_sup_total'] = 0
        total_log[str(i)]['pd_semantic_sup_num_ratio'] = 0

    n_logs = 0
    for file in os.listdir(path):
        if file.endswith('.json'):
            n_logs += 1
            with open(os.path.join(path, file), 'r') as file:
                log = json.load(file)

                if log['depth_diff'] > 0:
                    total_log_general['total']['depth_diff_num'] += 1
                    total_log_general['total']['depth_diff_total'] += log['depth_diff']

                if log['pd_not_leaves_trav_len'] > 0:
                    total_log_general['total']['pd_not_leaves_num'] += 1
                    total_log_general['total']['pd_not_leaves_total'] += log['pd_not_leaves_trav_len']

                if log['pd_spare_leaves_trav_len'] > 0:
                    total_log_general['total']['pd_spare_leaves_num'] += 1
                    total_log_general['total']['pd_spare_leaves_total'] += log['pd_spare_leaves_trav_len']

                for i in range(5):
                    if str(i) in log:
                        if log[str(i)]['n_children_diff'] > 0:
                            total_log[str(i)]['n_children_diff_num'] += 1
                            total_log[str(i)]['n_children_diff_total'] += log[str(i)]['n_children_diff']

                        if log[str(i)]['gt_semantic_sup'] > 0:
                            total_log[str(i)]['gt_semantic_sup_num'] += 1
                            total_log[str(i)]['gt_semantic_sup_total'] += log[str(i)]['gt_semantic_sup']

                        if log[str(i)]['pd_semantic_sup'] > 0:
                            total_log[str(i)]['pd_semantic_sup_num'] += 1
                            total_log[str(i)]['pd_semantic_sup_total'] += log[str(i)]['pd_semantic_sup']
    total_log_general['total']['n_objects'] = n_logs

    total_log_general['total']['depth_diff_num_ratio'] = total_log_general['total']['depth_diff_num'] / n_logs
    total_log_general['total']['pd_not_leaves_num_ratio'] = total_log_general['total']['pd_not_leaves_num'] / n_logs
    total_log_general['total']['pd_spare_leaves_num_ratio'] = total_log_general['total']['pd_spare_leaves_num'] / n_logs

    for i in range(5):
        total_log[str(i)]['n_children_diff_num_ratio'] = total_log[str(i)]['n_children_diff_num'] / n_logs
        total_log[str(i)]['gt_semantic_sup_num_ratio'] = total_log[str(i)]['gt_semantic_sup_num'] / n_logs
        total_log[str(i)]['pd_semantic_sup_num_ratio'] = total_log[str(i)]['pd_semantic_sup_num'] / n_logs

    return total_log, total_log_general

def iou(gt_nodes, pred_nodes, thr=0.5):
    smooth = 1e-6
    gt_mask, pred_mask = np.zeros((64, 64, 64)), np.zeros((64, 64, 64))
    for node in gt_nodes:
        geometry = np.where(node.geo.cpu().numpy()[0] > thr)
        gt_mask[geometry] = 1
    for node in pred_nodes:
        geometry = np.where(node.geo.cpu().numpy()[0][0] > thr)
        pred_mask[geometry] = 1
    intersection = gt_mask * pred_mask
    union = gt_mask + pred_mask - intersection
    metric = np.sum(intersection) / (np.sum(union) + smooth)
    return metric

def f1(gt_nodes, pred_nodes, thr=0.5):
    smooth = 1e-6
    gt_mask, pred_mask = np.zeros((64, 64, 64)), np.zeros((64, 64, 64))
    for node in gt_nodes:
        geometry = np.where(node.geo.cpu().numpy()[0] > thr)
        gt_mask[geometry] = 1
    for node in pred_nodes:
        geometry = np.where(node.geo.cpu().numpy()[0][0] > thr)
        pred_mask[geometry] = 1
    tp = gt_mask * pred_mask
    fp = pred_mask - tp
    fn = gt_mask - tp
    metric = tp.sum() / (tp.sum() + 0.5 * (fp.sum() + fn.sum()) + smooth)
    return metric

def grid_iou(gt_tree, pred_tree):
    grid = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    output_dict = {}
    output_dict['names'] = ['object', 'children', 'leaves']
    gt_leaves = [x for x in gt_tree.depth_first_traversal() if x.is_leaf]
    pred_leaves = [x for x in pred_tree.depth_first_traversal() if x.is_leaf]
    for thr in grid:
        object_iou = iou([gt_tree.root], [pred_tree.root], thr=thr)
        children_iou = iou(gt_tree.root.children, pred_tree.root.children, thr=thr)
        leaves_iou = iou(gt_leaves, pred_leaves, thr=thr)
        output_dict[thr] = [object_iou, children_iou, leaves_iou]
    return output_dict

def grid_f1(gt_tree, pred_tree):
    grid = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    output_dict = {}
    output_dict['names'] = ['object', 'children', 'leaves']
    gt_leaves = [x for x in gt_tree.depth_first_traversal() if x.is_leaf]
    pred_leaves = [x for x in pred_tree.depth_first_traversal() if x.is_leaf]
    for thr in grid:
        object_f1 = f1([gt_tree.root], [pred_tree.root], thr=thr)
        children_f1 = f1(gt_tree.root.children, pred_tree.root.children, thr=thr)
        leaves_f1 = f1(gt_leaves, pred_leaves, thr=thr)
        output_dict[thr] = [object_f1, children_f1, leaves_f1]
    return output_dict


def chamfer_distance(cloud_1, cloud_2):
    if len(cloud_1) == 0 or len(cloud_2) == 0:
        distance = 0
    else:
        dist_matrix = cdist(cloud_1, cloud_2)
        distance = dist_matrix.min(axis=0).sum() + dist_matrix.min(axis=1).sum()
    return distance


def make_assignment(gt_nodes, pred_nodes, thr):
    assignment = {}
    gt_nodes_with_labels = {}
    for i, node in enumerate(gt_nodes):
        if node.get_semantic_id() not in gt_nodes_with_labels:
            gt_nodes_with_labels[node.get_semantic_id()] = [i]
        else:
            gt_nodes_with_labels[node.get_semantic_id()] += [i]

    pred_nodes_with_labels = {}
    for i, node in enumerate(pred_nodes):
        if node.get_semantic_id() not in pred_nodes_with_labels:
            pred_nodes_with_labels[node.get_semantic_id()] = [i]
        else:
            pred_nodes_with_labels[node.get_semantic_id()] += [i]

    for gt_label in gt_nodes_with_labels:
        if gt_label not in pred_nodes_with_labels:
            for node in gt_nodes_with_labels[gt_label]:
                assignment[node] = -1
        else:
            if len(gt_nodes_with_labels[gt_label]) == 1 and len(pred_nodes_with_labels[gt_label]) == 1:
                assignment[gt_nodes_with_labels[gt_label][0]] = pred_nodes_with_labels[gt_label][0]
            else:
                dist_matrix = np.zeros((len(gt_nodes_with_labels[gt_label]), len(pred_nodes_with_labels[gt_label])))
                for i, gt_node in enumerate(gt_nodes_with_labels[gt_label]):
                    for j, pred_node in enumerate(pred_nodes_with_labels[gt_label]):
                        pc_1 = np.where(pred_nodes[pred_node].geo.cpu().numpy() > thr)
                        pc_1 = np.vstack([pc_1[2], pc_1[1], pc_1[0]]).T
                        pc_2 = np.where(gt_nodes[gt_node].geo.cpu().numpy())
                        pc_2 = np.vstack([pc_2[2], pc_2[1], pc_2[0]]).T
                        distance = chamfer_distance(pc_1, pc_2)
                        dist_matrix[i, j] = distance
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                for i in range(len(row_ind)):
                    assignment[gt_nodes_with_labels[gt_label][row_ind[i]]] = pred_nodes_with_labels[gt_label][
                        col_ind[i]]
    return assignment


def pc_iou(gt_nodes, pred_nodes, mesh, vox, thr=0.1):
    iou_final = {}

    g2w = vox.grid2world
    w2g = np.linalg.inv(g2w)
    surface_vertices = trimesh.sample.sample_surface(mesh, 12000)[0]
    transformed_vertices = np.hstack([surface_vertices, np.ones((len(surface_vertices), 1))])
    transformed_vertices = transformed_vertices @ w2g.T
    transformed_vertices = transformed_vertices[:, :3]
    transformed_vertices_int = transformed_vertices.astype('int')

    gt_mask, pred_mask = np.zeros((64, 64, 64)), np.zeros((64, 64, 64))
    for node in gt_nodes:
        geometry = np.where(node.geo.cpu().numpy()[0] > thr)
        gt_mask[geometry] = 1
    for node in pred_nodes:
        geometry = np.where(node.geo.cpu().numpy()[0][0] > thr)
        pred_mask[geometry] = 1

    gt_voxel_centers = np.where(gt_mask)
    gt_voxel_centers = np.vstack([gt_voxel_centers[2], gt_voxel_centers[1], gt_voxel_centers[0]]).T
    dist_matrix = cdist(gt_voxel_centers, transformed_vertices)
    arg_distances = dist_matrix.argmin(axis=0)
    point2voxel_map_gt = {i: arg_distances[i] for i in range(len(arg_distances))}
    voxel2point_map_gt = {}
    for point in point2voxel_map_gt:
        if point2voxel_map_gt[point] not in voxel2point_map_gt:
            voxel2point_map_gt[point2voxel_map_gt[point]] = [point]
        else:
            voxel2point_map_gt[point2voxel_map_gt[point]] += [point]
    voxel2point_map_explicit_gt = {}
    for voxel in voxel2point_map_gt:
        voxel2point_map_explicit_gt[tuple(gt_voxel_centers[voxel])] = voxel2point_map_gt[voxel]

    pd_voxel_centers = np.where(pred_mask)
    pd_voxel_centers = np.vstack([pd_voxel_centers[2], pd_voxel_centers[1], pd_voxel_centers[0]]).T
    dist_matrix = cdist(pd_voxel_centers, transformed_vertices)
    arg_distances = dist_matrix.argmin(axis=0)
    point2voxel_map_pd = {i: arg_distances[i] for i in range(len(arg_distances))}
    voxel2point_map_pd = {}
    for point in point2voxel_map_pd:
        if point2voxel_map_pd[point] not in voxel2point_map_pd:
            voxel2point_map_pd[point2voxel_map_pd[point]] = [point]
        else:
            voxel2point_map_pd[point2voxel_map_pd[point]] += [point]
    voxel2point_map_explicit_pd = {}
    for voxel in voxel2point_map_pd:
        voxel2point_map_explicit_pd[tuple(pd_voxel_centers[voxel])] = voxel2point_map_pd[voxel]

    assignment = make_assignment(gt_nodes, pred_nodes, thr)

    for gt_node_id in assignment:
        gt_node = gt_nodes[gt_node_id]
        pd_node = pred_nodes[assignment[gt_node_id]]

        gt_mask, pred_mask = np.zeros((64, 64, 64)), np.zeros((64, 64, 64))
        geometry = np.where(gt_node.geo.cpu().numpy()[0] > thr)
        gt_mask[geometry] = 1
        geometry = np.where(pd_node.geo.cpu().numpy()[0][0] > thr)
        pred_mask[geometry] = 1

        gt_voxel_centers = np.where(gt_mask)
        gt_voxel_centers = np.vstack([gt_voxel_centers[2], gt_voxel_centers[1], gt_voxel_centers[0]]).T
        pred_voxel_centers = np.where(pred_mask)
        pred_voxel_centers = np.vstack([pred_voxel_centers[2], pred_voxel_centers[1], pred_voxel_centers[0]]).T

        gt_point_mask = np.zeros((len(transformed_vertices)))
        pred_point_mask = np.zeros((len(transformed_vertices)))

        for gt_voxel_center in gt_voxel_centers:
            if tuple(gt_voxel_center) in voxel2point_map_explicit_gt:
                gt_point_mask[voxel2point_map_explicit_gt[tuple(gt_voxel_center)]] = 1
        for pred_voxel_center in pred_voxel_centers:
            if tuple(pred_voxel_center) in voxel2point_map_explicit_pd:
                pred_point_mask[voxel2point_map_explicit_pd[tuple(pred_voxel_center)]] = 1

        intersection = gt_point_mask * pred_point_mask
        union = gt_point_mask + pred_point_mask - intersection
        smooth = 1e-6
        metric = intersection.sum() / (union.sum() + smooth)

        if gt_node.get_semantic_id() not in iou_final:
            iou_final[gt_node.get_semantic_id()] = [metric]
        else:
            iou_final[gt_node.get_semantic_id()] += [metric]

    return iou_final