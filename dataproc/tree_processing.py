from copy import deepcopy
import numpy as np
from scipy.spatial.distance import cdist


def collect_leaves(hier, leaf_names, part_dir_names, leaf_name):
    if 'children' not in hier:
        leaf_names += [hier[leaf_name]]
        part_dir_names.extend(hier['objs'])
    else:
        for child in hier['children']:
            collect_leaves(child, leaf_names, part_dir_names, leaf_name)
    return leaf_names, part_dir_names


def collect_names(hier, node_names=None, node_full_names=None, leaf_name='name', full_name=''):
    if 'children' not in hier:
        node_names += [hier[leaf_name]]
    else:
        for child in hier['children']:
            node_names += [child[leaf_name]]
            node_full_names[child['dfs_id']] = '/'.join([full_name, child[leaf_name]])
            collect_names(child, node_names, node_full_names, leaf_name, '/'.join([full_name, child[leaf_name]]))
    return node_names, node_full_names


def propagate_leaves(hier, name_to_global_id):
    if 'children' not in hier:
        hier['full_global_id'] = []
        for obj in hier['objs']:
            if obj in name_to_global_id:
                hier['full_global_id'] += [int(name_to_global_id[obj])]
        return hier['full_global_id']
    else:
        hier['full_global_id'] = []
        for child in hier['children']:
            propagate_leaves(child, name_to_global_id)
            hier['full_global_id'].extend(child['full_global_id'])


def squeeze_tree(hier):
    if 'children' in hier:
        if len(hier['children']) == 1:
            if 'children' in hier['children'][0]:
                hier['children'] = hier['children'][0]['children']
                squeeze_tree(hier)
        else:
            for i, child in enumerate(hier['children']):
                if child['name'] not in ['chair_base']:
                    if 'children' in child:
                        if len(child['children']) == 1:
                            if 'children' in child['children'][0]:
                                child['children'] = child['children'][0]['children']
                                squeeze_tree(hier)
                        else:
                            squeeze_tree(child)
                else:
                    if 'children' in child:
                        if len(child['children']) == 1:
                            hier['children'][i] = child['children'][0]
                            squeeze_tree(hier)
                        else:
                            squeeze_tree(child)


def squeeze_the_only_leaves(hier):
    if 'children' in hier:
        if len(hier['children']) == 1 and ('children' not in hier['children'][0]):
            hier.pop('children', None)
        else:
            for child in hier['children']:
                squeeze_the_only_leaves(child)


def add_dfs_id(hier):
    dfs_id = 0

    def dfs(hier):
        nonlocal dfs_id
        hier['dfs_id'] = dfs_id
        dfs_id += 1
        if 'children' in hier:
            for child in hier['children']:
                dfs(child)

    dfs(hier)
    return dfs_id


def prepare_masks(hier, mask, num_nodes):
    masks = [None] * num_nodes

    def dfs(hier):
        nonlocal masks, mask
        dfs_id = hier['dfs_id']
        full_global_id = hier['full_global_id']
        masks[dfs_id] = np.isin(mask, full_global_id).astype('int')
        if 'children' in hier:
            for child in hier['children']:
                dfs(child)

    dfs(hier)
    return masks


def delete_nodes(hier, ids):
    if 'children' in hier:
        new_children = []
        for child in hier['children']:
            if child['dfs_id'] not in ids:
                new_children += [child]
        if len(new_children) != 0:
            hier['children'] = new_children
            for child in hier['children']:
                delete_nodes(child, ids)
        else:
            hier.pop('children', None)


def cut_tree(hier, valid_parts):
    if 'children' in hier:
        valid_children = []
        invalid_children = []
        for child in hier['children']:
            if child['name'] in valid_parts:
                valid_children += [child]
            else:
                invalid_children += [child]
        hier['children'] = valid_children

        if len(invalid_children) != 0:
            child_other = {}
            child_other['full_global_id'] = []
            child_other['name'] = 'other'
            child_other['Text'] = 'Other'
            for child in invalid_children:
                child_other['full_global_id'] += child['full_global_id']
            hier['children'] += [child_other]

        for child in hier['children']:
            cut_tree(child, valid_parts)


def leaf_ids(hier, ids):
    if 'children' in hier:
        for child in hier['children']:
            leaf_ids(child, ids)
    else:
        ids += [hier['dfs_id']]


def add_extra_children(hier):
    if 'children' in hier:
        children_global_ids = []
        for child in hier['children']:
            children_global_ids += child['full_global_id']
        extra_ids = list(set(hier['full_global_id']) - set(children_global_ids))
        if len(extra_ids) != 0:
            extra_child = {}
            extra_child['name'] = 'other'
            extra_child['Text'] = 'Other'
            extra_child['full_global_id'] = extra_ids
            hier['children'] += [extra_child]
        for child in hier['children']:
            add_extra_children(child)


def count_max_child(hier, num_children=None):
    if 'children' in hier:
        num_children += [len(hier['children'])]
        for child in hier['children']:
            count_max_child(child, num_children)


def detect_relationships(hier, masks):
    if 'children' in hier:

        # adjacency
        for i, child_i in enumerate(hier['children']):
            for j, child_j in enumerate(hier['children']):
                if i < j:
                    dfs_i = child_i['dfs_id']
                    dfs_j = child_j['dfs_id']
                    mask_i = masks[dfs_i]
                    mask_j = masks[dfs_j]
                    if mask_i.sum() == 0 or mask_j.sum() == 0:
                        continue
                    mask_i_coords = np.vstack(np.where(mask_i)[1:]).T
                    mask_j_coords = np.vstack(np.where(mask_j)[1:]).T
                    merged_coords = np.vstack([mask_i_coords, mask_j_coords])
                    min_dist = cdist(mask_i_coords, mask_j_coords).min()
                    if min_dist < 2:
                        if 'edges' not in hier:
                            hier['edges'] = [{"type": "ADJ", "part_a": i, "part_b": j}]
                        else:
                            hier['edges'] += [{"type": "ADJ", "part_a": i, "part_b": j}]

        # symmetry
        buckets = {}
        for i, child in enumerate(hier['children']):
            if child['name'] not in buckets:
                buckets[child['name']] = [(child['dfs_id'], i)]
            else:
                buckets[child['name']] += [(child['dfs_id'], i)]
        for key in buckets:
            if len(buckets[key]) > 1:
                for i, idx_pair_i in enumerate(buckets[key]):
                    for j, idx_pair_j in enumerate(buckets[key]):
                        if i < j:
                            mask_i = masks[idx_pair_i[0]]
                            mask_j = masks[idx_pair_j[0]]
                            if mask_i.sum() == 0 or mask_j.sum() == 0:
                                continue
                            if np.abs(mask_i.sum() - mask_j.sum()) / max(mask_i.sum(), mask_j.sum()) < 0.3:
                                if 'edges' not in hier:
                                    hier['edges'] = [{"type": "SYM", "part_a": idx_pair_i[1], "part_b": idx_pair_j[1]}]
                                else:
                                    hier['edges'] += [{"type": "SYM", "part_a": idx_pair_i[1], "part_b": idx_pair_j[1]}]

        for child in hier['children']:
            detect_relationships(child, masks)


def random_leaf(hier, children_path=None):
    if 'children' in hier:
        random_child_idx = np.random.randint(len(hier['children']))
        children_path += [random_child_idx]
        random_leaf(hier['children'][random_child_idx], children_path)


def remove_leaf(hier, children_path, chosen_child=None):
    if len(children_path) > 1:
        remove_leaf(hier['children'][children_path[0]], children_path[1:], chosen_child)
    elif len(children_path) == 1:
        new_children = []
        for i, child in enumerate(hier['children']):
            if i != children_path[0]:
                new_children += [child]
            else:
                chosen_child += [child['dfs_id']]
        hier['children'] = new_children


def remove_random_leaf(hier, proba_thrs=None):
    uniform_sample = np.random.rand()
    new_hier = deepcopy(hier)
    dfs_ids = []
    for proba_thr in proba_thrs:
        if uniform_sample < proba_thr:
            children_path = []
            chosen_child = []
            random_leaf(new_hier, children_path)
            remove_leaf(new_hier, children_path, chosen_child)
            if chosen_child[0] is not None:
                dfs_ids += [chosen_child[0]]
    return new_hier, dfs_ids


def fix_global_id(hier):
    if 'children' not in hier:
        return hier['full_global_id']
    else:
        new_global_id = []
        for child in hier['children']:
            new_global_id += fix_global_id(child)
        hier['full_global_id'] = new_global_id
        if len(hier['full_global_id']) == 1:
            return hier['full_global_id']
        else:
            return hier['full_global_id']


def add_complete_dfs_id(hier):
    hier['complete_dfs_id'] = hier['dfs_id']
    if 'children' in hier:
        for child in hier['children']:
            add_complete_dfs_id(child)


def fix_masks(masks, removed_masks):
    new_masks = []
    for mask in masks:
        for remove_mask in removed_masks:
            masks_intersection = mask * remove_mask
            mask = mask - masks_intersection
        new_masks += [mask]
    return new_masks


def squeeze_empty_full_global_ids(hier):
    if 'children' in hier:
        new_children = []
        for child in hier['children']:
            if len(child['full_global_id']) != 0:
                new_children += [child]
        hier['children'] = new_children
        for child in hier['children']:
            squeeze_empty_full_global_ids(child)


def replace_nodes(hier, replace_dict):
    if hier['name'] in replace_dict:
        hier['name'] = replace_dict[hier['name']]
    if 'children' in hier:
        for child in hier['children']:
            if child['name'] in replace_dict:
                child['name'] = replace_dict[child['name']]
        for child in hier['children']:
            replace_nodes(child, replace_dict)


def merge_instances(hier):
    if 'children' in hier:
        new_children = []
        children_instances = {}
        for child in hier['children']:
            if child['name'] not in children_instances:
                children_instances[child['name']] = [child]
            else:
                children_instances[child['name']] += [child]
        for child_name in children_instances:
            if len(children_instances[child_name]) > 1 and child_name != 'chair_arm':
                new_child = {}
                new_child['name'] = child_name
                new_child['text'] = children_instances[child_name][0]['text']
                new_child['full_global_id'] = []
                for old_child in children_instances[child_name]:
                    new_child['full_global_id'] += old_child['full_global_id']
                new_children += [new_child]
            else:
                new_children += children_instances[child_name]
        hier['children'] = new_children
        for child in hier['children']:
            merge_instances(child)


def merge_instances_full(hier):
    if 'children' in hier:
        new_children = []
        children_instances = {}
        for child in hier['children']:
            if child['name'] not in children_instances:
                children_instances[child['name']] = [child]
            else:
                children_instances[child['name']] += [child]
        for child_name in children_instances:
            if len(children_instances[child_name]) > 1:
                new_child = {}
                new_child['name'] = child_name
                new_child['full_global_id'] = []
                new_child_children = []
                for old_child in children_instances[child_name]:
                    new_child['full_global_id'] += old_child['full_global_id']
                    if 'children' in old_child:
                        new_child_children += old_child['children']
                if len(new_child_children) != 0:
                    new_child['children'] = new_child_children
                new_children += [new_child]
            else:
                new_children += children_instances[child_name]
        hier['children'] = new_children
        for child in hier['children']:
            merge_instances_full(child)


def keep_first_children(hier):
    if 'children' in hier:
        for child in hier['children']:
            if 'children' in child:
                child.pop('children')


def unmerge_chair_arms(hier, masks, size):
    if 'children' in hier:
        for i, child in enumerate(hier['children']):
            if child['name'] == 'chair_arm':
                arm_dfs = child['dfs_id']
                coords = np.stack(np.where(masks[arm_dfs])).T
                if coords.mean(axis=0)[3] < size / 2:
                    hier['children'][i]['name'] = 'chair_arm_left'
                else:
                    hier['children'][i]['name'] = 'chair_arm_right'


def squeeze_parts(hier, part_names):
    flag = False
    if 'children' in hier:
        new_children = []
        for i, child in enumerate(hier['children']):
            if child['name'] in part_names:
                flag = True
                if 'children' in child:
                    for cchild in child['children']:
                        new_children += [cchild]
                else:
                    if child['name'] not in ['tabletop']:
                        child['name'] = 'other'
                    else:
                        child['name'] = 'tabletop_single'
                    new_children += [child]
            else:
                new_children += [child]
        hier['children'] = new_children
        if flag:
            squeeze_parts(hier, part_names)
        else:
            for child in hier['children']:
                squeeze_parts(child, part_names)