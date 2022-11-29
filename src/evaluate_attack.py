import os
import torch
import numpy as np
import matplotlib.pylab as plt
import pickle
from utils import visualize_mesh_subplots, get_directories, \
                  get_device, log_string, get_file, get_argument_parser
from utils_attack import get_attack_params, get_attack_weights, get_results_path
from plotly_visualization import visualize_and_compare
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import time
import datetime
from curvature import meancurvature_diff_abs
from matplotlib.ticker import FormatStrFormatter

import matplotlib as mpl

# ---------- Letex font ----------
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'

debug_mode = False
flags = get_argument_parser()

# ---------- Configurations ----------
if debug_mode:
    rand_seed = 1
    dataset = 'coma'
    evaluation_type = 'visual'
    result_type = 'saga'
    visualize = True
    show_src_heatmap = False
    save_images = False
    image_suffix = '.png'
else:
    rand_seed = flags.seed
    dataset = flags.dataset
    evaluation_type = flags.evaluation_type
    result_type = flags.result_type
    visualize = True if evaluation_type == 'visual' else flags.visualize
    show_src_heatmap = flags.show_src_heatmap
    save_images = flags.save_images
    image_suffix = flags.image_suffix

torch.manual_seed(rand_seed)
params = get_attack_params(dataset)
weights = get_attack_weights(dataset)
data_dir, _, logs_dir, results_dir, images_dir = get_directories(dataset=dataset)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = get_file(logs_dir, 'evaluate_attack_' + dataset + '_' + evaluation_type + '_' + current_time + '.txt')
device = get_device(log_file=log_file)

results_file = get_results_path(dataset, results_dir, result_type)
log_string(log_file, 'result_type: {}\nresults file: {}'.format(result_type, results_file))
with open(results_file, 'rb') as handle:
    all_pair_dict_list = pickle.load(handle)

# ---------- Considering only the end-of-attack savings ----------
pair_dict_list = [item for item in all_pair_dict_list if
                  (item["step"] == (params["N_attack_steps"] - 1))]

save_faces_file = os.path.join(data_dir, 'raw', 'mesh_faces.npy')
assert os.path.exists(save_faces_file), 'saved faces file was not found'
mesh_faces = np.load(save_faces_file)

log_string(log_file, 'num_pairs: {}'.format(len(pair_dict_list)))

# ---------- Visual Results ----------
if evaluation_type == 'visual':
    vis_according_class = True
    if vis_according_class:
        if dataset == 'coma':
            vis_pairs = [item for item in pair_dict_list if (item["s_label"] == 4 and item["t_label"] == 9)]
        else:  # dataset == 'smal'
            vis_pairs = [item for item in pair_dict_list if (item["s_label"] == 4 and item["t_label"] == 0)]
    else:
        if dataset == 'coma':
            pair_numbers = [4784, 4220, 2268, 443, 2188, 1609, 1651, 4034]
        else:  # dataset == 'smal'
            pair_numbers = [374, 828, 458, 852, 260, 513]
        vis_pairs = [item for item in pair_dict_list if (item["pair_number"] in pair_numbers)]

    for pair_dict in vis_pairs:
        pair_number = pair_dict["pair_number"]
        s_mesh = pair_dict["s_mesh"]
        t_mesh = pair_dict["t_mesh"]
        adv_mesh = pair_dict["adv_mesh"]
        s_recon_mesh = pair_dict["s_recon_mesh"]
        t_recon_mesh = pair_dict["t_recon_mesh"]
        adv_recon_mesh = pair_dict["adv_recon_mesh"]

        if visualize:
            save_s_image = os.path.join(images_dir, 'src_' + str(pair_number) + image_suffix)
            save_t_image = os.path.join(images_dir, 'target_' + str(pair_number) + image_suffix)
            save_adv_image = os.path.join(images_dir, 'adv_' + str(pair_number) + image_suffix)
            save_adv_recon_image = os.path.join(images_dir, 'adv_recon_' + str(pair_number) + image_suffix)

            visualize_mesh_subplots(s_mesh, adv_mesh, mesh_faces, title_1='src_' + str(pair_number),
                                    title_2='adv_' + str(pair_number), save_file_1=save_s_image,
                                    save_file_2=save_adv_image)
            visualize_mesh_subplots(t_mesh, adv_recon_mesh, mesh_faces, title_1='target_' + str(pair_number),
                                    title_2='adv_recon_' + str(pair_number), save_file_1=save_t_image,
                                    save_file_2=save_adv_recon_image)

            if show_src_heatmap:
                src_v = torch.from_numpy(s_mesh)
                adv_v = torch.from_numpy(adv_mesh)
                faces_v = torch.from_numpy(mesh_faces)
                visualize_and_compare(adv_v, faces_v, src_v, faces_v,
                                      (src_v - adv_v).norm(p=2, dim=-1))

# ---------- Spectral Analysis ----------
elif evaluation_type == 'beta':
    save_beta_freq_image = os.path.join(images_dir, dataset + '_beta' + image_suffix)

    add_beta_values = [item["add_beta"] for item in pair_dict_list]

    num_shapes = len(add_beta_values)
    num_freq = add_beta_values[0].shape[0]
    add_beta = np.array(add_beta_values)
    add_beta_scalar_per_vertex = np.linalg.norm(add_beta, axis=-1)
    add_beta = np.mean(add_beta_scalar_per_vertex, axis=0)
    frequencies = np.array([np.int(x) for x in range(num_freq)])

    s_alpha_values = [item["s_alpha"] for item in pair_dict_list]
    s_alpha = np.linalg.norm(np.array(s_alpha_values), axis=-1)
    s_alpha_mean = np.mean(s_alpha, axis=0)[:len(add_beta)]

    title_prefix = 'Mean '
    graph_title = dataset + ' ' + title_prefix + 'Additive Beta vs. Frequency'
    fig, ax1 = plt.subplots()
    ax1.plot(frequencies, add_beta, color='red', lw=2, label=r'$\bar{\beta}$')
    ax1.plot(frequencies, s_alpha_mean, color='blue', lw=2, label=r'$\bar{\alpha}$')
    ax1.legend(loc="upper right", fontsize=20)
    y_lim_factor = 12 if dataset == 'coma' else 2.4
    ax1.set_xlim(0, 100)
    ax1.tick_params(axis='x', labelsize=20)
    y_limit = s_alpha_mean.max() / y_lim_factor
    ax1.set_ylim(0, y_limit)

    ax1.set_xlabel('Frequency', fontsize=20)
    ax1.set_ylabel('Magnitude', fontsize=20)
    ax1.set_yticks([])
    #ax1.tick_params(axis='y', labelcolor='blue')

    #fig.suptitle(graph_title, fontsize=14)

    if save_images:
        fig.savefig(save_beta_freq_image, bbox_inches='tight')
    else:
        plt.show()

# ---------- Mean Curvature Distortion ----------
elif evaluation_type == 'curv_dist':
    adv_t_curv_err_list = []
    pert_curv_err_list = []
    recon_curv_err_list = []
    pert_curv_norm_err_list = []
    recon_curv_norm_err_list = []
    for pair in pair_dict_list:
        s_mesh = pair["s_mesh"]
        adv_mesh = pair["adv_mesh"]
        t_mesh = pair["t_mesh"]
        s_recon_mesh = pair["s_recon_mesh"]
        adv_recon_mesh = pair["adv_recon_mesh"]
        t_recon_mesh = pair["t_recon_mesh"]

        #  convert to torch
        s_mesh_torch = torch.from_numpy(s_mesh).to(device)
        adv_mesh_torch = torch.from_numpy(adv_mesh).to(device)
        t_mesh_torch = torch.from_numpy(t_mesh).to(device)
        s_recon_mesh_torch = torch.from_numpy(s_recon_mesh).to(device)
        adv_recon_mesh_torch = torch.from_numpy(adv_recon_mesh).to(device)
        t_recon_mesh_torch = torch.from_numpy(t_recon_mesh).to(device)
        mesh_faces_torch = torch.from_numpy(mesh_faces).to(device)

        #  convert to torch
        curv_time_start = time.time()
        abs_curv_err_adv = meancurvature_diff_abs(adv_mesh_torch, s_mesh_torch, mesh_faces_torch)
        abs_curv_err_adv_recon = meancurvature_diff_abs(adv_recon_mesh_torch, t_mesh_torch, mesh_faces_torch)
        log_string(log_file, 'Curvature calculation time: {} seconds'
                   .format(time.time() - curv_time_start))
        log_string(log_file, 'curv_adv: {}, curv_adv_recon: {}'
                   .format(abs_curv_err_adv, abs_curv_err_adv_recon))

        numerical_error_threshold = 1000
        if np.any(np.isnan([abs_curv_err_adv, abs_curv_err_adv_recon])) or \
           np.any(np.array([abs_curv_err_adv, abs_curv_err_adv_recon]) > numerical_error_threshold):
            log_string(log_file, ' extreme curvature problem, details: pair: {}, curv_adv: {}, curv_adv_recon: {},'
                       .format(pair['pair_number'], abs_curv_err_adv, abs_curv_err_adv_recon))
        else:
            pert_curv_err_list.append(abs_curv_err_adv)
            recon_curv_err_list.append(abs_curv_err_adv_recon)

    log_string(log_file, 'max_values of curvature: {}'.format(np.array(pert_curv_err_list).max()))
    pert_curv_err_mean = np.mean(pert_curv_err_list)
    recon_curv_err_mean = np.mean(recon_curv_err_list)
    log_string(log_file, 'pert_curv_err_mean: {}, recon_curv_err_mean: {},'
               .format(pert_curv_err_mean, recon_curv_err_mean))

# ---------- Frequency Ablation Study (different number of eigenvectors) ----------
elif evaluation_type == 'freq_ablation':
    save_freq_ablation = os.path.join(images_dir, dataset + '_freq_ablation' + image_suffix)

    """
    This evaluation requires prior experiments.
    Please run the attack multiple times with a different number of used eigenvectors.
    Then, save the curvature distortion values in .pickle file and name it: "freq_ablation.pickle"
    The values should be saved in following dictionary: {'delta_s': [list], 'delta_t': [list], 'x_axis': [list]}
    """

    freq_ablation_file = os.path.join(results_dir, 'freq_ablation.pickle')
    with open(freq_ablation_file, 'rb') as h:
        freq_ablation_dict = pickle.load(h)
    adv_curv_values = freq_ablation_dict['delta_s']
    adv_recon_curv_values = freq_ablation_dict['delta_t']
    num_evects_values = freq_ablation_dict['x_axis']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax1.plot(num_evects_values, adv_curv_values, color='blue', marker='o')
    ax2.plot(num_evects_values, adv_recon_curv_values, color='red', marker='o')
    y_max = max(adv_curv_values + adv_recon_curv_values)
    vertical_x = 500 if dataset == 'coma' else 2000
    vertical_y = (adv_curv_values[num_evects_values.index(500)] / y_max) if dataset == 'coma' else \
        (adv_recon_curv_values[num_evects_values.index(2000)] / y_max)
    ax3.axvline(x=vertical_x, ymin=0, ymax=vertical_y, linestyle='--', color='black')
    ax3.axis('off')

    ax1.set_xticks([500, 1000, 1500, 2000, 2500, 3000])
    ax1.tick_params(axis='both', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.set_xlabel('Number of Eigenvectors', fontsize=20)
    ax1.set_ylabel(r'$\bar{\delta}_{\mathcal{S}}$', color='blue', rotation='horizontal', fontsize=25, labelpad=25)
    ax2.set_ylabel(r'$\bar{\delta}_{\mathcal{T}}$', color='red', rotation='horizontal', fontsize=25)
    ax2.yaxis.set_label_coords(1.20, 0.62)

    if save_images:
        fig.savefig(save_freq_ablation, bbox_inches='tight')
    else:
        plt.show()

# ---------- t-SNE - Latent Space Analysis ----------
elif evaluation_type == 'tsne':
    save_tsne_image = os.path.join(images_dir, dataset + '_tsne' + image_suffix)

    t_latent_values = [item["t_latent"] for item in pair_dict_list]
    t_label_values = [item["t_label"].detach().cpu().numpy() for item in pair_dict_list]

    shapes_per_class_tsne = 50
    num_shapes_for_tsne = shapes_per_class_tsne * 11 if dataset == 'coma' else shapes_per_class_tsne * 5
    num_shapes = len(t_latent_values)
    tsne_idxs = np.random.randint(low=0, high=(num_shapes - 1), size=num_shapes_for_tsne)
    t_labels_for_tsne = np.array(t_label_values)[tsne_idxs]
    if dataset == 'coma':
        num_adv_shapes = 4
        adv_latent_9_1 = [item["adv_latent"] for item in pair_dict_list if
                          (item["s_label"] == 9 and item["t_label"] == 1)]
        adv_latent_9_1 = np.squeeze(np.array(adv_latent_9_1))[:num_adv_shapes]
        adv_9_to_1_labels = np.array(['9 to 1' for i in range(num_adv_shapes)])

        adv_latent_10_5 = [item["adv_latent"] for item in pair_dict_list if
                           (item["s_label"] == 10 and item["t_label"] == 5)]
        adv_latent_10_5 = np.squeeze(np.array(adv_latent_10_5))[:num_adv_shapes]
        adv_10_to_5_labels = np.array(['10 to 5' for i in range(num_adv_shapes)])

        labels_for_tsne = np.concatenate((t_labels_for_tsne, adv_9_to_1_labels, adv_10_to_5_labels), axis=-1)
        hue_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '9 to 1', '10 to 5']
        palette = {'0': 'tab:blue', '1': 'tab:green', '2': 'tab:orange', '3': 'tab:red', '4': 'tab:cyan', '5': 'gold',
                   '6': 'tab:pink', '7': 'tab:purple', '8': 'tab:brown', '9': 'tab:olive', '10': 'magenta',
                   '9 to 1': 'black', '10 to 5': 'dimgray'}
        markers = {'0': 'o', '1': 'o', '2': 'o', '3': 'o', '4': 'o', '5': 'o', '6': 'o',
                   '7': 'o', '8': 'o', '9': 'o', '10': 'o', '9 to 1': 'X', '10 to 5': 'X'}
        markers_sizes = {'0': 50, '1': 50, '2': 50, '3': 50, '4': 50, '5': 50, '6': 50,
                         '7': 50, '8': 50, '9': 50, '10': 50, '9 to 1': 200, '10 to 5': 200}

        t_latent = np.array(t_latent_values)[tsne_idxs].squeeze()
        latent = np.concatenate((t_latent, adv_latent_9_1, adv_latent_10_5), axis=0)
        perplexity = 30.0

    else:
        num_adv_shapes = 2
        adv_latent_2_1 = [item["adv_latent"] for item in pair_dict_list if
                          (item["pair_number"] == 417 or item["pair_number"] == 425)]
        adv_latent_2_1 = np.squeeze(np.array(adv_latent_2_1))[:num_adv_shapes]
        adv_2_to_1_labels = np.array(['2 to 1' for i in range(num_adv_shapes)])

        adv_latent_4_0 = [item["adv_latent"] for item in pair_dict_list if
                          (item["pair_number"] == 804 or item["pair_number"] == 852)]
        adv_latent_4_0 = np.squeeze(np.array(adv_latent_4_0))[:num_adv_shapes]
        adv_4_to_0_labels = np.array(['4 to 0' for i in range(num_adv_shapes)])

        labels_for_tsne = np.concatenate((t_labels_for_tsne, adv_2_to_1_labels, adv_4_to_0_labels), axis=-1)
        hue_order = ['0', '1', '2', '3', '4', '2 to 1', '4 to 0']
        palette = {'0': 'tab:blue', '1': 'tab:green', '2': 'tab:orange', '3': 'tab:red', '4': 'tab:cyan',
                   '2 to 1': 'black', '4 to 0': 'dimgray'}
        markers = {'0': 'o', '1': 'o', '2': 'o', '3': 'o', '4': 'o', '2 to 1': 'X', '4 to 0': 'X'}
        markers_sizes = {'0': 50, '1': 50, '2': 50, '3': 50, '4': 50, '2 to 1': 200, '4 to 0': 200}

        t_latent = np.array(t_latent_values)[tsne_idxs].squeeze()
        latent = np.concatenate((t_latent, adv_latent_2_1, adv_latent_4_0), axis=0)
        perplexity = 50.0

    latent_embedded = TSNE(n_components=2, init='random', perplexity=perplexity).fit_transform(latent)

    df = pd.DataFrame()
    df['x_axis'] = latent_embedded[:, 0]
    df['y_axis'] = latent_embedded[:, 1]
    df['Identity'] = labels_for_tsne

    sns.scatterplot(data=df, x='x_axis', y='y_axis', hue='Identity', style='Identity', size='Identity',
                    hue_order=hue_order, palette=palette, sizes=markers_sizes, markers=markers)
    # plt.title(dataset + ' Latent Space', weight='bold').set_fontsize('14')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Identities', fontsize='large',
               title_fontsize='large')
    if save_images:
        plt.savefig(save_tsne_image, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
