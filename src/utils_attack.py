import os
import time
import torch
import numpy as np
from utils import log_string, visualize_mesh_subplots
from utils_data import convert_class_name_to_label
from plotly_visualization import visualize_and_compare
import matplotlib.pylab as plt


def get_barycenter_matrix(n, faces):
    # calc_adj_matrix
    A = torch.zeros((n, n))
    A[faces[:, 0], faces[:, 1]] = 1
    A[faces[:, 1], faces[:, 2]] = 1
    A[faces[:, 2], faces[:, 0]] = 1

    A = torch.matmul(torch.diag(1 / torch.sum(A, dim=1)), A)
    bary = (A - torch.eye(n))

    return bary


def get_attack_weights(dataset):
    weights = {
        'W_recon_mse': 1,  # Weight of the perturbation loss
        'W_reg_spat': 0,
        'W_reg_bary': 100 if dataset == 'coma' else 50,
        'W_reg_edge': 2 if dataset == 'coma' else 5,
        'W_reg_area': 500 if dataset == 'coma' else 0,
        'W_reg_normals': 0 if dataset == 'coma' else 0.5,
        'W_reg_chamfer': 0,
    }
    return weights


def get_attack_params(dataset):
    if dataset == 'coma':
        class_name_list = ['person_1', 'person_2', 'person_3', 'person_4', 'person_5', 'person_6',
                           'person_7', 'person_8', 'person_9', 'person_10', 'person_11', 'person_12']
        classes_for_source = ['person_1', 'person_2', 'person_3', 'person_4', 'person_5', 'person_6',
                              'person_7', 'person_8', 'person_9', 'person_10', 'person_11']
        classes_for_target = ['person_1', 'person_2', 'person_3', 'person_4', 'person_5', 'person_6',
                              'person_7', 'person_8', 'person_9', 'person_10', 'person_11']
    else:  # dataset == 'smal'
        class_name_list = ['cat', 'cow', 'dog', 'hippo', 'horse']
        classes_for_source = ['cat', 'cow', 'dog', 'hippo', 'horse']
        classes_for_target = ['cat', 'cow', 'dog', 'hippo', 'horse']

    params = {  # General Attack parameters
        'reduced_memory_mode': 'weak',  # ('strong' 'weak' 'none')
        'purpose': 'test',  # ('train' 'val' 'test')
        'L_rate_attack': 0.0001 if dataset == 'coma' else 0.01,  # Learning rate for the attack
        'N_attack_steps': 500 if dataset == 'coma' else 3000,  # Number of gradient steps for the attack
        'B_size': 1,  # Batch Size
        'N_evals': 500 if dataset == 'coma' else 2000,  # Number of eigenvalues
        'N_evects': 500 if dataset == 'coma' else 2000,  # Number of eigenvalues
        'Adversary_type': 'beta',  # ('beta', 'delta')
        'pert_type': 'add' if dataset == 'coma' else 'mul',  # ('add', 'mul') additive or multiplicative perturbations
        'weights_on_evects': 'low',  # ('low', 'high')
        # Attack modes
        'class_name_list': class_name_list,
        'classes_for_source': classes_for_source,
        'classes_for_target': classes_for_target,
        'num_src_per_class': 50,
        'num_targets_per_src': 1,
        'random_targets_mode': False,
        'inter_class_pairs': True,
        'use_self_evects': False,
    }
    return params


def get_results_file_name(seed, dataset, params, weights):
    file_name = dataset \
                + '___seed_' + str(seed) \
                + '___adv_type_' + str(params['Adversary_type']) \
                + '___self_evects_' + str(params['use_self_evects']) \
                + '___random_targets_' + str(params['random_targets_mode']) \
                + '___N_evects_' + str(params['N_evects']) \
                + '___W_bary_' + str(weights['W_reg_bary']) \
                + '___W_edge_' + str(weights['W_reg_edge']) \
                + '___W_area_' + str(weights['W_reg_area']) \
                + '___W_normals_' + str(weights['W_reg_normals']) \
                + '___W_chamfer_' + str(weights['W_reg_chamfer']) \
                + '.pickle'
    return file_name


def prepare_data_for_attack(params, source_classes_for_attack, data, slice_idx,
                            class_idx_matrix, nn_idx_matrix, device, log_file=None):
    meshes = data.get_np_poses()
    alphas = data.get_np_alphas()

    class_name_list = params['class_name_list']
    target_classes_for_attack = params['classes_for_target']
    num_targets_per_src = params['num_targets_per_src']
    inter_class_pairs = params['inter_class_pairs']

    num_classes = len(class_name_list)
    source_mesh_list = []
    source_alphas_list = []

    target_mesh_list = []
    target_alphas_list = []

    for i in range(num_classes):
        source_class_name = class_name_list[i]
        if source_class_name not in source_classes_for_attack:
            continue

        source_attack_idx = class_idx_matrix[i]
        num_sources_for_attack = len(source_attack_idx)
        log_string(log_file, "choosing to attack class: {}, class indices: {}, num_sources_for_attack: {}".format(
            source_class_name, source_attack_idx, num_sources_for_attack))

        # get all the class data
        source_class_mesh = meshes[slice_idx[i]:slice_idx[i + 1]]
        source_class_alphas = alphas[slice_idx[i]:slice_idx[i + 1]]

        # from the class data, extract the indices for attack according to the class-idx matrix
        source_class_mesh_for_attack = source_class_mesh[source_attack_idx]
        source_class_alphas_for_attack = source_class_alphas[source_attack_idx]

        if not inter_class_pairs:
            log_string(log_file, "intra-class attack: choosing targets from src class: {}".format(source_class_name))

            # In this mode we pick the targets to be other shapes from the src class
            # Unlike the inter-class mode, each src is tested with all the target shapes chosen.
            target_attack_idx_range = [idx for idx in range(slice_idx[i], slice_idx[i + 1])]
            target_attack_idx = np.random.choice(target_attack_idx_range, size=num_targets_per_src, replace=False)
            log_string(log_file, "target indices: {}".format(target_attack_idx))

            target_mesh_for_attack = meshes[target_attack_idx]
            target_alphas_for_attack = alphas[target_attack_idx]

            source_mesh_list.append(source_class_mesh_for_attack)
            source_alphas_list.append(source_class_alphas_for_attack)

            target_mesh_list.append(target_mesh_for_attack)
            target_alphas_list.append(target_alphas_for_attack)

            break

        target_mesh_for_attack_list = []
        target_alphas_for_attack_list = []

        for j in range(num_classes):
            target_class_name = class_name_list[j]
            if target_class_name not in target_classes_for_attack or target_class_name == source_class_name:
                continue

            nn_idx_s_class_t_class = nn_idx_matrix[slice_idx[i]:slice_idx[i + 1], slice_idx[j]:slice_idx[j + 1]]
            nn_idx_s_for_attack_t_class = nn_idx_s_class_t_class[source_attack_idx].copy()

            target_class_mesh = meshes[slice_idx[j]:slice_idx[j + 1]]
            target_class_alphas = alphas[slice_idx[j]:slice_idx[j + 1]]

            target_class_mesh_for_attack_list = []
            target_class_alphas_for_attack_list = []
            for s in range(num_sources_for_attack):
                target_attack_idx = nn_idx_s_for_attack_t_class[s, :num_targets_per_src]
                log_string(log_file,
                           "for shape number: {} out of {} src-shapes to attack in src-class: {}, the target-class is: {}, choose targets: {} when number of targets per src is: {}"
                           .format(s, num_sources_for_attack, i, j, target_attack_idx, num_targets_per_src))
                target_class_mesh_for_attack_curr = target_class_mesh[target_attack_idx]
                target_class_alphas_for_attack_curr = target_class_alphas[target_attack_idx]

                target_class_mesh_for_attack_list.append(np.expand_dims(target_class_mesh_for_attack_curr, axis=0))
                target_class_alphas_for_attack_list.append(np.expand_dims(target_class_alphas_for_attack_curr, axis=0))

            # shape after vstack = [num_src_per_class ; num_targets_per_src ; vertices ; axes]
            target_mesh_for_attack = np.vstack(target_class_mesh_for_attack_list)
            target_alphas_for_attack = np.vstack(target_class_alphas_for_attack_list)

            target_mesh_for_attack_list.append(target_mesh_for_attack)
            target_alphas_for_attack_list.append(target_alphas_for_attack)

        # shape after concatenate = [num_src_per_class ; (num_targets_per_src * num_classes) ; vertices ; axes]
        target_mesh_for_attack_concat = np.concatenate(target_mesh_for_attack_list, axis=1)
        target_alphas_for_attack_concat = np.concatenate(target_alphas_for_attack_list, axis=1)

        old_shape_mesh = target_mesh_for_attack_concat.shape
        new_shape_mesh = [old_shape_mesh[0] * old_shape_mesh[1]] + [old_shape_mesh[n] for n in
                                                                    range(2, len(old_shape_mesh))]
        target_mesh_curr = np.reshape(target_mesh_for_attack_concat, new_shape_mesh)

        old_shape_alphas = target_alphas_for_attack_concat.shape
        new_shape_alphas = [old_shape_alphas[0] * old_shape_alphas[1]] + [old_shape_alphas[n] for n in
                                                                          range(2, len(old_shape_alphas))]
        target_alphas_curr = np.reshape(target_alphas_for_attack_concat, new_shape_alphas)

        # new shape = [(num_tgts_per_src * num_classes * num_src_per_class) ; vertices ; axes]
        target_mesh_list.append(target_mesh_curr)
        target_alphas_list.append(target_alphas_curr)

        source_mesh_list.append(source_class_mesh_for_attack)
        source_alphas_list.append(source_class_alphas_for_attack)

    source_mesh = np.vstack(source_mesh_list)
    source_alphas = np.vstack(source_alphas_list)

    target_mesh = np.vstack(target_mesh_list)
    target_alphas = np.vstack(target_alphas_list)

    # Convert to torch
    source_mesh = torch.from_numpy(source_mesh.astype(np.float32)).to(device)
    target_mesh = torch.from_numpy(target_mesh.astype(np.float32)).to(device)
    source_alphas = torch.from_numpy(source_alphas.astype(np.float32)).to(device)
    target_alphas = torch.from_numpy(target_alphas.astype(np.float32)).to(device)

    return source_mesh, source_alphas, target_mesh, target_alphas


def evaluate_and_save_attack(step, pair_counter, pair_dict_list, dataset, device, src_class, target_class,
                             params, weights, AE, mesh_faces, s_mesh, s_evects, s_alphas, t_mesh, t_evects,
                             t_alphas, adv, recon_losses, reg_losses, visualize=False, log_file=None):
    show_src_heatmap = False
    mesh_faces_np = mesh_faces.detach().cpu().numpy()
    # Losses
    [loss_recon_total_per_step, loss_recon_mse_per_step] = recon_losses

    [loss_reg_total_per_step, loss_reg_spat_per_step, loss_reg_bary_per_step, loss_reg_edge_per_step,
     loss_reg_area_per_step, loss_reg_normals_per_step, loss_reg_chamfer_per_step] = reg_losses

    if visualize:
        # ---------- plot the loss terms during training ----------
        plt.figure('Reconstruction losses vs. Attack Step')
        plt.title('Reconstruction losses vs. Attack Step')

        plt.plot(loss_recon_total_per_step, 'red', label='loss_recon_total_per_step')
        if weights['W_recon_mse']: plt.plot(loss_recon_mse_per_step, 'green', label='loss_recon_mse_per_step')

        plt.legend()
        plt.xlabel('Attack Step')
        plt.ylabel('Loss Value')
        plt.show()

        plt.figure('Regularization losses vs. Attack Step')
        plt.title('Regularization losses vs. Attack Step')

        plt.plot(loss_reg_total_per_step, 'red', label='loss_reg_total_per_step')
        if weights['W_reg_spat']: plt.plot(loss_reg_spat_per_step, 'green', label='loss_reg_spat_per_step')
        if weights['W_reg_bary']: plt.plot(loss_reg_bary_per_step, 'black', label='loss_reg_bary_per_step')
        if weights['W_reg_edge']: plt.plot(loss_reg_edge_per_step, 'blue', label='loss_reg_edge_per_step')
        if weights['W_reg_area']: plt.plot(loss_reg_area_per_step, 'magenta', label='loss_reg_area_per_step')
        if weights['W_reg_normals']: plt.plot(loss_reg_normals_per_step, 'cyan', label='loss_reg_normals_per_step')
        if weights['W_reg_chamfer']: plt.plot(loss_reg_chamfer_per_step, 'green', label='loss_reg_chamfer_per_step')

        plt.legend()
        plt.xlabel('Attack Step')
        plt.ylabel('Loss Value')
        plt.show()

    ############ final source-perturbation and target-reconstruction ############
    adv_mesh = adv.spatial_attack(params, s_mesh, evectors=s_evects, alphas=s_alphas)
    adv_recon_mesh, adv_latent = AE(adv_mesh)

    ############ priginal source and target reconstructions #####################
    s_recon_mesh, s_latent = AE(s_mesh)
    t_recon_mesh, t_latent = AE(t_mesh)

    ############ alignments and numpy conversions ###############################
    s_mesh_np = s_mesh.detach().cpu().numpy()
    s_recon_mesh_np = s_recon_mesh.detach().cpu().numpy()
    s_latent_np = s_latent.detach().cpu().numpy()
    t_mesh_np = t_mesh.detach().cpu().numpy()
    t_recon_mesh_np = t_recon_mesh.detach().cpu().numpy()
    t_latent_np = t_latent.detach().cpu().numpy()
    adv_mesh_np = adv_mesh.detach().cpu().numpy()
    adv_recon_mesh_np = adv_recon_mesh.detach().cpu().numpy()
    adv_latent_np = adv_latent.detach().cpu().numpy()

    ############ plot visualizations ############################################
    if visualize:
        visualize_mesh_subplots(s_mesh_np[0], adv_mesh_np[0], mesh_faces_np)
        visualize_mesh_subplots(t_mesh_np[0], adv_recon_mesh_np[0], mesh_faces_np)

        if show_src_heatmap:
            visualize_and_compare(adv_mesh[0], mesh_faces, s_mesh[0], mesh_faces,
                                  (s_mesh[0] - adv_mesh[0]).norm(p=2, dim=-1))

    ############ save src-target pair information ###############################
    s_label = convert_class_name_to_label(classname=src_class, dataset=dataset, device=device)
    t_label = convert_class_name_to_label(classname=target_class, dataset=dataset, device=device)
    if params['Adversary_type'] == 'beta':
        beta_np = adv.get_pert_beta()[0].detach().cpu().numpy()
        s_alpha_np = s_alphas[0].detach().cpu().numpy()
        t_alpha_np = t_alphas[0].detach().cpu().numpy()
        adv_alpha_np = torch.linalg.lstsq(s_evects[0], adv_mesh[0])[0].detach().cpu().numpy()
        adv_recon_alpha_np = torch.linalg.lstsq(t_evects[0], adv_recon_mesh[0])[0].detach().cpu().numpy()
        num_pert_evects = len(beta_np)
        num_tot_evects = len(s_alpha_np)

        s_alpha_np_tmp = s_alpha_np[:num_pert_evects] if (params['weights_on_evects'] == 'low') \
            else s_alpha_np[(num_tot_evects - num_pert_evects):]
        if params['pert_type'] == 'add':
            add_beta_np = beta_np
        else:  # params['pert_type'] == 'mul'
            add_beta_np = beta_np * s_alpha_np_tmp
        s_label = convert_class_name_to_label(classname=src_class, dataset=dataset, device=device)
        t_label = convert_class_name_to_label(classname=target_class, dataset=dataset, device=device)
        pair_dict = {"step": step, "pair_number": pair_counter, "s_label": s_label, "t_label": t_label,
                     "s_mesh": s_mesh_np[0], "t_mesh": t_mesh_np[0],
                     "adv_mesh": adv_mesh_np[0], "adv_recon_mesh": adv_recon_mesh_np[0],
                     "s_recon_mesh": s_recon_mesh_np[0], "t_recon_mesh": t_recon_mesh_np[0],
                     "s_alpha": s_alpha_np, "t_alpha": t_alpha_np,
                     "adv_alpha": adv_alpha_np, "adv_recon_alpha": adv_recon_alpha_np,
                     "s_latent": s_latent_np, "t_latent": t_latent_np, "adv_latent": adv_latent_np,
                     "add_beta": add_beta_np}
    else:  # params['Adversary_type'] == 'delta'
        pair_dict = {"step": step, "pair_number": pair_counter, "s_label": s_label, "t_label": t_label,
                     "s_mesh": s_mesh_np[0], "t_mesh": t_mesh_np[0],
                     "adv_mesh": adv_mesh_np[0], "adv_recon_mesh": adv_recon_mesh_np[0],
                     "s_recon_mesh": s_recon_mesh_np[0], "t_recon_mesh": t_recon_mesh_np[0],
                     "s_latent": s_latent_np, "t_latent": t_latent_np, "adv_latent": adv_latent_np}

    if params['inter_class_pairs']:
        log_string(log_file, "saving pair_number: {}, s_label: {}, t_label: {}".format(
            pair_dict["pair_number"], pair_dict["s_label"], pair_dict["t_label"]))
    else:
        log_string(log_file, "intra-class mode - saving pair_number: {}, s_label: {}".format(pair_dict["pair_number"],
                                                                                             pair_dict["s_label"]))
    pair_dict_list.append(pair_dict)

    return pair_dict_list


def run_attack(pair_counter, pair_dict_list, dataset, device, src_class, target_class,
               params, weights, AE, mesh_faces, s_mesh, s_evects, s_alphas,
               t_mesh, t_evects, t_alphas, adv, evaluate_every, log_file=None, visualize=False):
    start_single_attack_time = time.time()

    adv.init_pert(s_mesh.size(dim=0), mesh_faces)
    optimizer = get_attack_optimizers(params, adv)

    loss_recon_mse_per_step = []

    loss_reg_spat_per_step = []
    loss_reg_bary_per_step = []
    loss_reg_edge_per_step = []
    loss_reg_area_per_step = []
    loss_reg_normals_per_step = []
    loss_reg_chamfer_per_step = []

    loss_recon_total_per_step = []
    loss_reg_total_per_step = []
    loss_total_per_step = []

    for step in range(0, params['N_attack_steps']):
        log_string(log_file, "Attack step number: {}".format(step))

        adv_mesh = adv.spatial_attack(params, s_mesh, evectors=s_evects, alphas=s_alphas)
        adv_recon_mesh, _ = AE(adv_mesh)

        # ------------------------------ Reconstruction Loss ------------------------------
        loss_recon_mse = adv.get_mse_loss(weights, t_mesh, adv_recon_mesh)
        loss_recon_total = loss_recon_mse

        # ------------------------------ Perturbation Loss ------------------------------
        if params['Adversary_type'] == 'beta':
            loss_reg_spat = adv.get_beta_loss(weights)
        else:  # delta
            loss_reg_spat = adv.get_delta_loss(weights)

        loss_reg_bary = adv.get_barycenter_loss(weights, s_mesh, adv_mesh)
        loss_reg_edge = adv.get_edge_loss(weights, s_mesh, adv_mesh, mesh_faces)
        loss_reg_area = adv.get_area_loss(weights, s_mesh, adv_mesh, mesh_faces)
        loss_reg_normals = adv.get_normals_loss(weights, s_mesh, adv_mesh, mesh_faces)
        loss_reg_chamfer = adv.get_chamfer_loss(weights, s_mesh, adv_mesh)

        loss_reg_total = loss_reg_spat + loss_reg_bary + loss_reg_edge + loss_reg_area + loss_reg_normals + loss_reg_chamfer

        # ------------------------------ Total Loss ------------------------------
        loss_total = loss_recon_total + loss_reg_total
        loss_total.backward()

        # ------------------------------ Optimizer Step ------------------------------
        optimizer.step()
        optimizer.zero_grad()

        # ------------------------------ Save Losses ------------------------------
        loss_recon_total_per_step.append(loss_recon_total.item())
        loss_reg_total_per_step.append(loss_reg_total.item())
        loss_total_per_step.append(loss_total.item())

        loss_recon_mse_per_step.append(loss_recon_mse.item())

        loss_reg_spat_per_step.append(loss_reg_spat.item())
        loss_reg_bary_per_step.append(loss_reg_bary.item())
        loss_reg_edge_per_step.append(loss_reg_edge.item())
        loss_reg_area_per_step.append(loss_reg_area.item())
        loss_reg_normals_per_step.append(loss_reg_normals.item())
        loss_reg_chamfer_per_step.append(loss_reg_chamfer.item())

        if (step + 1) % evaluate_every == 0:
            recon_losses = [loss_recon_total_per_step, loss_recon_mse_per_step]
            reg_losses = [loss_reg_total_per_step, loss_reg_spat_per_step, loss_reg_bary_per_step,
                          loss_reg_edge_per_step, loss_reg_area_per_step, loss_reg_normals_per_step,
                          loss_reg_chamfer_per_step]
            pair_dict_list = evaluate_and_save_attack(step, pair_counter, pair_dict_list, dataset, device, src_class,
                                                      target_class,
                                                      params, weights, AE, mesh_faces, s_mesh, s_evects, s_alphas,
                                                      t_mesh, t_evects, t_alphas, adv, recon_losses, reg_losses,
                                                      visualize=visualize, log_file=log_file)

    end_single_attack_time = time.time()
    single_attack_duration = (end_single_attack_time - start_single_attack_time) / 60
    log_string(log_file, 'Single Attack duration: %.2f minutes: ' % single_attack_duration)

    return pair_dict_list


def get_attack_optimizers(params, adv):
    if params['Adversary_type'] == 'beta':
        optimizer = torch.optim.Adam([adv.get_pert_beta()], lr=params['L_rate_attack'])
    else:  # params['Adversary_type'] == 'delta'
        optimizer = torch.optim.Adam([adv.get_pert_delta()], lr=params['L_rate_attack'])
    return optimizer


def apply_flags_to_params(flags, params, weights):
    params['reduced_memory_mode'] = flags.reduced_memory_mode
    params['purpose'] = flags.purpose
    params['L_rate_attack'] = flags.learning_rate
    params['N_attack_steps'] = flags.attack_steps
    params['B_size'] = flags.attack_batch_size
    params['N_evals'] = flags.num_evals
    params['N_evects'] = flags.num_evects
    params['Adversary_type'] = flags.adversary_type
    params['pert_type'] = flags.pert_type
    params['weights_on_evects'] = flags.weights_on_evects
    params['num_src_per_class'] = flags.num_src_per_class
    params['num_targets_per_src'] = flags.num_targets_per_src
    params['random_targets_mode'] = flags.random_targets_mode
    params['inter_class_pairs'] = flags.inter_class_pairs
    params['use_self_evects'] = flags.use_self_evects

    weights['W_recon_mse'] = flags.w_recon_mse
    weights['W_reg_spat'] = flags.w_reg_spat
    weights['W_reg_bary'] = flags.w_reg_bary
    weights['W_reg_edge'] = flags.w_reg_edge
    weights['W_reg_area'] = flags.w_reg_area
    weights['W_reg_normals'] = flags.w_reg_normals
    weights['W_reg_chamfer'] = flags.w_reg_chamfer

    return params


def get_results_path(dataset, results_dir, result_type):
    if dataset == 'coma':
        if result_type == 'saga':
            result_file_name = ''
        elif result_type == 'pc':
            result_file_name = ''
        elif result_type == 'delta':
            result_file_name = ''
        elif result_type == 'oods':
            result_file_name = ''
        elif result_type == 'oodt':
            result_file_name = ''
        elif result_type == 'coma_transfer':
            result_file_name = ''
        elif result_type == 'mlp_transfer':
            result_file_name = ''
        elif result_type == 'self_evects':
            result_file_name = ''
        elif result_type == 'random_targets':
            result_file_name = ''
        else:
            NotImplementedError()
    else:  # dataset == "smal"
        if result_type == 'saga':
            result_file_name = ''
        elif result_type == 'pc':
            result_file_name = ''
        elif result_type == 'delta':
            result_file_name = ''
        elif result_type == 'self_evects':
            result_file_name = ''
        elif result_type == 'random_targets':
            result_file_name = ''
        else:
            NotImplementedError()

    results_file = os.path.join(results_dir, result_file_name)
    assert os.path.exists(results_file), 'results file was not found'

    return results_file
