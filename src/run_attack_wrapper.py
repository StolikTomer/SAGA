import os
import torch
import numpy as np
import time
import datetime
import pickle
from adversary import Adversary
from utils import log_string, get_device, get_file, visualize_mesh_subplots, get_directories, get_argument_parser
from utils_data import load_data, get_slice_idx
from utils_models import load_trained_ae, get_ae_params
from utils_attack import get_attack_params, get_attack_weights, prepare_data_for_attack, run_attack,\
                         get_results_file_name, apply_flags_to_params
from spectral_torch import calculate_eigenvalues_batch

debug_mode = False
flags = get_argument_parser()

dataset = flags.dataset
params = get_attack_params(dataset=dataset)
weights = get_attack_weights(dataset=dataset)
ae_params = get_ae_params()

if debug_mode:
    seed = 1
    visualize = True
    show_src_heatmap = False
    save_results = False
    if dataset == 'coma':
        params['classes_for_source'] = ['person_5']
        params['classes_for_target'] = ['person_10']
    else:  # dataset == 'smal'
        params['classes_for_source'] = ['horse']
        params['classes_for_target'] = ['cat']
else:
    params = apply_flags_to_params(flags, params, weights)
    seed = flags.seed
    visualize = flags.visualize
    show_src_heatmap = flags.show_src_heatmap
    save_results = flags.save_results

# ---------- Paths ----------
data_dir, models_dir, logs_dir, results_dir, _ = get_directories(dataset=dataset)
autoencoder_dir = os.path.join(models_dir, 'autoencoders')

# ---------- Configurations ----------
purpose = params['purpose']
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = get_file(logs_dir, 'run_attack_wrapper_' + dataset + '_' + current_time + '.txt')
log_string(log_file, "debug_mode = {}".format(debug_mode))
log_string(log_file, "params:\n\n{}\n\n".format(params))
log_string(log_file, "weights:\n\n{}\n\n".format(weights))
device = get_device(log_file=log_file)
results_file_name = get_results_file_name(seed, dataset, params, weights)
log_string(log_file, "results_file_name:\n\n{}\n\n".format(results_file_name))
results_file = os.path.join(results_dir, results_file_name)

torch.manual_seed(seed)
log_string(log_file, "seed = {}".format(seed))

# ---------- Load data ----------
data_struct = load_data(dataset=dataset, params=params, data_dir=data_dir,
                        device=device, log_file=log_file)
num_shapes = data_struct.get_num_shapes()
num_vertices = data_struct.get_num_vertices()
num_dims = data_struct.get_pos(0).shape[-1]
shared_evects = data_struct.get_evects()
mesh_faces = data_struct.get_faces()
mesh_faces_np = mesh_faces.detach().cpu().numpy()

# ---------- Load the source\target idx matrix ----------
index_info_dir = os.path.join(data_dir, 'index')
src_matrix_file = os.path.join(index_info_dir, purpose + '_src_idx_matrix.npy')
l2_nn_idx_file = os.path.join(index_info_dir, purpose + '_random_l2_nn_idx.npy') if params['random_targets_mode'] \
                 else os.path.join(index_info_dir, purpose + '_sorted_l2_nn_idx.npy')

assert os.path.exists(src_matrix_file), 'src_matrix_file does not exist, please run prepare_indices_for_attack'
assert os.path.exists(l2_nn_idx_file), 'l2_nn_idx_file does not exist, please run prepare_indices_for_attack'

src_matrix_idx = np.load(src_matrix_file)
src_matrix_idx = src_matrix_idx[:, :params['num_src_per_class']]
l2_nn_idx = np.load(l2_nn_idx_file)

# ---------- Load Model ----------
input_shape = [num_vertices, num_dims]
AE = load_trained_ae(ae_params, autoencoder_dir, device, input_shape)

# ---------- Create an Adversary ----------
adv = Adversary(params, num_vertices, device=device, log_file=log_file)

class_name_list = params['class_name_list']
classes_for_source = params['classes_for_source']
num_src_per_class = params['num_src_per_class']
num_targets_per_src = params['num_targets_per_src']
slice_idx = get_slice_idx(dataset=dataset, params=params)

# Attack the AE model
pair_dict_list = []
pair_counter = 0

log_string(log_file, 'start general attack ...')
start_attack_time = time.time()
for i in range(len(class_name_list)):
    class_name = class_name_list[i]
    if class_name not in classes_for_source:
        continue
    if not params['inter_class_pairs']:
        num_target_classes = 1
    else:
        classes_for_target = params['classes_for_target'].copy()
        if class_name in classes_for_target:
            classes_for_target.remove(class_name)
        num_target_classes = len(classes_for_target)

    # prepare data for attack
    log_string(log_file, 'prepare data for attack ...')
    source_mesh, source_alphas, target_mesh, target_alphas = \
        prepare_data_for_attack(params, [class_name], data_struct, slice_idx,
                                src_matrix_idx, l2_nn_idx, device, log_file=log_file)

    log_string(log_file, "start attack for src-class: {}".format(class_name))

    for src_idx in range(len(source_mesh)):
        if params['Adversary_type'] == 'beta':
            s_mesh = source_mesh[src_idx].unsqueeze(dim=0)
            source_evects = None
            if params['use_self_evects']:
                _, source_evects = \
                    calculate_eigenvalues_batch(s_mesh, mesh_faces, params['N_evals'], device, (num_vertices - 2),
                                                log_file=log_file, return_vectors=True)
                if source_evects is None:
                    continue

                s_evects = source_evects
                s_alphas = torch.linalg.lstsq(s_evects[0], s_mesh[0])[0].unsqueeze(dim=0)
            else:
                s_evects = shared_evects.unsqueeze(dim=0)
                s_alphas = source_alphas[src_idx].unsqueeze(dim=0)

        else:
            s_mesh = source_mesh[src_idx].unsqueeze(dim=0)
            s_alphas = None
            s_evects = None

        num_all_targets_per_src = num_targets_per_src * num_target_classes
        for j in range(num_target_classes):
            for k in range(num_targets_per_src):
                if params['inter_class_pairs']:
                    target_index = (src_idx * num_all_targets_per_src) + (j * num_targets_per_src) + k
                else:
                    target_index = k

                if params['Adversary_type'] == 'beta':
                    t_mesh = target_mesh[target_index].unsqueeze(dim=0)
                    t_evects = shared_evects.unsqueeze(dim=0)
                    t_alphas = target_alphas[target_index].unsqueeze(dim=0)
                else:
                    t_mesh = target_mesh[target_index].unsqueeze(dim=0)
                    t_evects = None
                    t_alphas = None

                if visualize:
                    s_recon, _ = AE(s_mesh)
                    t_recon, _ = AE(t_mesh)

                    s_mesh_np = s_mesh.detach().cpu().numpy()
                    t_mesh_np = t_mesh.detach().cpu().numpy()
                    s_recon_np = s_recon.detach().cpu().numpy()
                    t_recon_np = t_recon.detach().cpu().numpy()

                    visualize_mesh_subplots(s_mesh_np[0], s_recon_np[0], mesh_faces_np)
                    visualize_mesh_subplots(t_mesh_np[0], t_recon_np[0], mesh_faces_np)

                src_class = class_name
                target_class = classes_for_target[j] if params["inter_class_pairs"] else class_name
                pair_dict_list = run_attack(pair_counter, pair_dict_list, dataset, device, src_class, target_class,
                                            params, weights, AE, mesh_faces, s_mesh, s_evects, s_alphas,
                                            t_mesh, t_evects, t_alphas, adv, params['N_attack_steps'],
                                            log_file=log_file, visualize=visualize)
                pair_counter = pair_counter + 1

end_attack_time = time.time()
attack_duration = (end_attack_time - start_attack_time) / 60
log_string(log_file, 'Total Attack duration: %.2f minutes' % attack_duration)

save_faces_file = os.path.join(data_dir, 'raw', 'mesh_faces.npy')
if save_results:
    with open(results_file, 'wb') as handle:
        pickle.dump(pair_dict_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    np.save(save_faces_file, mesh_faces_np)

log_string(log_file, 'end attack ...')
