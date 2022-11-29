import os
import numpy as np
import pickle
import torch
import datetime

from utils import get_argument_parser, visualize_mesh_subplots, get_directories, get_device, log_string, get_file
from utils_attack import get_attack_params, get_attack_weights, get_results_path
from train_classifier import evaluate
from utils_models import load_trained_classifier, get_classifier_params, load_trained_ae, get_ae_params

flags = get_argument_parser()

dataset = flags.dataset
dataset = 'smal'
visualize = flags.visualize
save_images = flags.save_images
result_type = flags.result_type
print_confusion_matrix = flags.print_confusion_matrix
seed = flags.seed
image_suffix = flags.image_suffix
stability_step = flags.stability_step
stability_step = 3

ae_params = get_ae_params()
cls_params = get_classifier_params()
attack_params = get_attack_params(dataset=dataset)
attack_weights = get_attack_weights(dataset=dataset)
data_dir, models_dir, logs_dir, results_dir, images_dir = get_directories(dataset=dataset)
autoencoder_dir = os.path.join(models_dir, 'autoencoders')
classifier_dir = os.path.join(models_dir, 'classifiers')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = get_file(logs_dir, 'adversarial_stability_' + current_time + '.txt')
device = get_device(log_file=log_file)

if stability_step == 0:
    results_file = get_results_path(dataset, results_dir, result_type)
elif stability_step == 1:
    results_file = os.path.join(results_dir, dataset + '_target_stability_ae_step_1.pickle')
elif stability_step == 2:
    results_file = os.path.join(results_dir, dataset + '_target_stability_ae_step_2.pickle')
elif stability_step == 3:
    results_file = os.path.join(results_dir, dataset + '_target_stability_ae_step_3.pickle')

assert os.path.exists(results_file), 'adversarial_stability - the requested results file path does not exist:\n{}'.format(results_file)
log_string(log_file, 'result_type: {}\nresults file: {}'.format(result_type, results_file))

with open(results_file, 'rb') as handle:
    all_pair_dict_list = pickle.load(handle)

# considering only the end-of-attack savings
pair_dict_list = [item for item in all_pair_dict_list if
                  (item["step"] == (attack_params["N_attack_steps"] - 1))]

log_string(log_file, 'num_pairs: {}'.format(len(pair_dict_list)))

save_faces_file = os.path.join(data_dir, 'raw', 'mesh_faces.npy')
assert os.path.exists(save_faces_file), 'saved faces file was not found'
mesh_faces = np.load(save_faces_file)

num_vertices = pair_dict_list[0]['s_mesh'].shape[0]
num_dims = pair_dict_list[0]['s_mesh'].shape[1]
input_shape = [num_vertices, num_dims]

AE = load_trained_ae(ae_params, autoencoder_dir, device, input_shape)

dict_out_list = []
for pair_dict in pair_dict_list:
    step = pair_dict['step']
    pair_number = pair_dict['pair_number']

    adv_mesh_np = pair_dict['adv_mesh']
    s_mesh_np = pair_dict['s_mesh']
    t_mesh_np = pair_dict['t_mesh']
    t_label = pair_dict['t_label']

    adv_mesh = torch.from_numpy(adv_mesh_np).unsqueeze(dim=0).to(device)
    t_mesh = torch.from_numpy(t_mesh_np).unsqueeze(dim=0).to(device)

    with torch.no_grad():
        adv_recon_mesh, _ = AE(adv_mesh)
        t_recon_mesh, _ = AE(t_mesh)

    adv_recon_mesh_np = adv_recon_mesh[0].detach().cpu().numpy()
    t_recon_mesh_np = t_recon_mesh[0].detach().cpu().numpy()

    if dataset == 'coma':
        vis_numbers = [4784, 4220, 2268, 443, 2188, 1609, 1651, 4034]
    else:  # dataset == 'smal'
        vis_numbers = [374, 828, 458, 852, 260, 513]
    if visualize and pair_number in vis_numbers:
        visualize_mesh_subplots(adv_mesh_np, adv_recon_mesh_np, mesh_faces, title_1='adv_input_' + str(pair_number),
                                title_2='adv_recon_' + str(pair_number))
        visualize_mesh_subplots(t_mesh_np, adv_recon_mesh_np, mesh_faces, title_1='t_input_' + str(pair_number),
                                title_2='t_recon_' + str(pair_number))

    dict_out = {'step': step, 'pair_number': pair_number,
                's_mesh': s_mesh_np, 't_mesh': t_recon_mesh_np,
                'adv_mesh': adv_recon_mesh_np, 't_label': t_label}
    dict_out_list.append(dict_out)

t_label_list = [item['t_label'] for item in dict_out_list]
adv_recon_mesh_list = [torch.from_numpy(item['adv_mesh']).to(device) for item in dict_out_list]
t_recon_mesh_list = [torch.from_numpy(item['t_mesh']).to(device) for item in dict_out_list]


# ---------- Loading the pre-trained classifier ----------
classifier = load_trained_classifier(cls_params, classifier_dir, device, dataset)

adv_recon_accuracy, adv_recon_confusion = evaluate(adv_recon_mesh_list, t_label_list, classifier, faces=mesh_faces, log_file=log_file)
log_string(log_file, '\nAdv-Recon classification accuracy: {}'.format(adv_recon_accuracy))
if print_confusion_matrix:
    log_string(log_file, '\nAdv-Recon classification confusion:\n{}\n'.format(adv_recon_confusion))

t_recon_accuracy, t_recon_confusion = evaluate(t_recon_mesh_list, t_label_list, classifier, faces=mesh_faces, log_file=log_file)
log_string(log_file, '\nTarget-Recon classification accuracy: {}'.format(t_recon_accuracy))
if print_confusion_matrix:
    log_string(log_file, '\nTarget-Recon classification confusion:\n{}\n'.format(t_recon_confusion))

results_file_out = os.path.join(results_dir, dataset + '_target_stability_ae_step_' + str(stability_step + 1) + '.pickle')
log_string(log_file, 'saving out dictionary, results_file_out: {}'.format(results_file_out))
with open(results_file_out, 'wb') as handle:
    pickle.dump(dict_out_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
