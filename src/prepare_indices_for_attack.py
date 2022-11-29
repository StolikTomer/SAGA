import os
import numpy as np
import time
import datetime
from utils import get_file, log_string, get_device, get_directories, get_argument_parser
from utils_data import load_data, get_slice_idx
from utils_attack import get_attack_params


def save_src_idx_matrix(params, num_classes, slice_idx, seed, src_matrix_file):
    sel_idx = -1 * np.ones([num_classes, params['num_src_per_class']], dtype=np.int16)
    for i in range(num_classes):
        np.random.seed(seed)

        num_examples = slice_idx[i + 1] - slice_idx[i]
        perm = np.arange(num_examples)
        np.random.shuffle(perm)

        num_instances = min(params['num_src_per_class'], num_examples)
        sel_idx[i, :num_instances] = perm[:params['num_src_per_class']]

    np.save(src_matrix_file, sel_idx)


def get_l2_nn(meshes, num_classes, start_idx, batch_size, l2_dist_mat_file,
              l2_nn_idx_file, log_file, random_mode):
    start_time = time.time()
    num_examples_all, num_points, _ = meshes.shape
    l2_batch_size = 10

    # compute l2 distance matrix
    meshes_curr = meshes[start_idx:(start_idx + batch_size)]
    num_examples_curr = len(meshes_curr)
    l2_dist_mat_curr = -1 * np.ones([num_examples_all, num_examples_curr], dtype=np.float32)

    source_mesh = np.tile(np.expand_dims(meshes_curr, axis=0), [num_examples_all, 1, 1, 1])
    target_mesh = np.tile(np.expand_dims(meshes, axis=1), [1, num_examples_curr, 1, 1])

    for i in range(0, num_examples_all, l2_batch_size):
        for j in range(0, num_examples_curr, l2_batch_size):
            sources = source_mesh[i:i + l2_batch_size, j:j + l2_batch_size]
            targets = target_mesh[i:i + l2_batch_size, j:j + l2_batch_size]

            s_batch = np.reshape(sources, [-1, num_points, 3])
            t_batch = np.reshape(targets, [-1, num_points, 3])
            dist_batch = np.mean(np.linalg.norm((s_batch - t_batch), axis=-1), axis=-1)
            dist_batch_reshape = np.reshape(dist_batch, sources.shape[:2])
            l2_dist_mat_curr[i:i + l2_batch_size, j:j + l2_batch_size] = dist_batch_reshape

    assert l2_dist_mat_curr.min() >= 0, 'the l2_dist_mat_curr matrix was not filled correctly'

    # save current l2 distance data
    if os.path.exists(l2_dist_mat_file):
        l2_dist_mat = np.load(l2_dist_mat_file)
    else:
        l2_dist_mat = -1 * np.ones([num_examples_all, num_examples_all], dtype=np.float32)

    l2_dist_mat[:, start_idx:(start_idx + batch_size)] = l2_dist_mat_curr
    np.save(l2_dist_mat_file, l2_dist_mat)

    duration = time.time() - start_time
    log_string(log_file, 'start index %d end index %d, out of size %d, duration (minutes): %.2f' %
               (start_idx, min(start_idx + batch_size, num_examples_all), num_examples_all,
                duration / 60.0))

    if l2_dist_mat.min() >= 0:
        # nearest neighbors indices
        l2_nn_idx = sort_dist_mat(l2_dist_mat, num_classes, random_mode)

        np.save(l2_nn_idx_file, l2_nn_idx)


def sort_dist_mat(dist_mat, num_classes, random_mode):
    nn_idx = -1 * np.ones(dist_mat.shape, dtype=np.int16)

    # sorting indices (in ascending order) for each pair of source and target classes. Note:
    # 1. the indices start from 0 for each pair
    # 2. for same source and target classes (intra class), for each instance the smallest distance is 0. thus, the first index should be discarded
    for i in range(num_classes):
        for j in range(num_classes):
            dist_mat_source_target = dist_mat[slice_idx[i]:slice_idx[i + 1], slice_idx[j]:slice_idx[j + 1]]
            if random_mode:
                np.apply_along_axis(np.random.shuffle, axis=1, arr=dist_mat_source_target)
                sort_idx = dist_mat_source_target.astype(np.int16)
            else:
                sort_idx = np.argsort(dist_mat_source_target, axis=1).astype(np.int16)
            nn_idx[slice_idx[i]:slice_idx[i + 1], slice_idx[j]:slice_idx[j + 1]] = sort_idx

    assert nn_idx.min() >= 0, 'the nn_idx matrix was not filled correctly'
    return nn_idx


if __name__ == "__main__":
    flags = get_argument_parser()

    dataset = flags.dataset
    params = get_attack_params(dataset=dataset)
    params['purpose'] = 'test'
    params['random_targets_mode'] = flags.random_targets_mode
    params['reduced_memory_mode'] = flags.reduced_memory_mode
    params['num_src_per_class'] = flags.num_src_per_class
    seed = flags.seed

    data_dir, _, logs_dir, _, _ = get_directories(dataset=dataset)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = get_file(logs_dir, 'prepare_indices_for_attack' + current_time + '.txt')
    log_string(log_file, "seed = {}\nparams:\n\n{}\n\n".format(seed, params))
    device = get_device(log_file=log_file)
    data_struct = load_data(dataset=dataset, params=params, data_dir=data_dir, device=device, log_file=log_file)
    slice_idx = get_slice_idx(dataset=dataset, params=params)
    num_classes = len(slice_idx) - 1

    index_info_dir = os.path.join(data_dir, 'index')
    if not os.path.exists(index_info_dir):
        os.mkdir(index_info_dir)

    ##################
    # random indices #
    ##################
    src_idx_matrix_file = os.path.join(index_info_dir, params['purpose'] + '_src_idx_matrix.npy')
    save_src_idx_matrix(params, num_classes, slice_idx, seed, src_idx_matrix_file)

    #############################################################
    # Nearest neighbors matrix calculation #
    #############################################################
    meshes_np = data_struct.get_np_poses()
    num_shapes = data_struct.get_num_shapes()

    l2_dist_mat_file = os.path.join(index_info_dir, params['purpose'] + '_l2_dist_mat.npy')
    if params['random_targets_mode']:
        l2_nn_idx_file = os.path.join(index_info_dir, params['purpose'] + '_random_l2_nn_idx.npy')
    else:
        l2_nn_idx_file = os.path.join(index_info_dir, params['purpose'] + '_sorted_l2_nn_idx.npy')
    batch_size = 100
    for start_idx in range(0, num_shapes, batch_size):
        log_string(log_file,
                   "calling get_l2_nn, start_idx={}, out of num_shapes={}"
                   .format(start_idx, num_shapes))
        get_l2_nn(meshes_np, num_classes, start_idx, batch_size, l2_dist_mat_file,
                  l2_nn_idx_file, log_file, params['random_targets_mode'])

    #############################################################
    # Check Results #
    #############################################################
    class_idx_matrix = np.load(src_idx_matrix_file)
    l2_dist_mat = np.load(l2_dist_mat_file)
    l2_dist_mat = np.load(l2_nn_idx_file)

    print(class_idx_matrix.shape)
    print(l2_dist_mat.shape)
