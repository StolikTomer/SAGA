import os
import numpy as np
import pickle
import torch
from utils import visualize_mesh_subplots, get_directories, get_device, log_string, get_file, get_argument_parser
from utils_attack import get_attack_params, get_attack_weights, get_results_path
from utils_models import load_trained_ae, get_ae_params
import datetime

flags = get_argument_parser()

dataset = flags.dataset
visualize = flags.visualize
result_type = flags.result_type

ae_params = get_ae_params()
params = get_attack_params(dataset=dataset)
weights = get_attack_weights(dataset=dataset)
data_dir, models_dir, logs_dir, results_dir, images_dir = get_directories(dataset=dataset)
autoencoder_dir = os.path.join(models_dir, 'autoencoders')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = get_file(logs_dir, 'transfer_attack_' + current_time + '.txt')
device = get_device(log_file=log_file)

save_faces_file = os.path.join(data_dir, 'raw', 'mesh_faces.npy')
assert os.path.exists(save_faces_file), 'saved faces file was not found'

results_file = get_results_path(dataset, results_dir, result_type)
log_string(log_file, 'result_type: {}\nresults file: {}'.format(result_type, results_file))
with open(results_file, 'rb') as handle:
    all_pair_dict_list = pickle.load(handle)

# considering only the end-of-attack savings
pair_dict_list = [item for item in all_pair_dict_list if
                  (item["step"] == (params["N_attack_steps"] - 1))]

log_string(log_file, 'num_pairs: {}'.format(len(pair_dict_list)))

mesh_faces = np.load(save_faces_file)

num_vertices = pair_dict_list[0]['s_mesh'].shape[0]
num_dims = pair_dict_list[0]['s_mesh'].shape[1]
input_shape = [num_vertices, num_dims]

AE = load_trained_ae(ae_params, autoencoder_dir, device, input_shape, ae_dir_name='transfer_mlp')

dict_out_list = []
for pair_dict in pair_dict_list:
    pair_number = pair_dict["pair_number"]
    s_mesh_np = pair_dict["s_mesh"]
    t_mesh_np = pair_dict["t_mesh"]
    adv_mesh_np = pair_dict["adv_mesh"]
    step = pair_dict["step"]
    s_label = pair_dict["s_label"]
    t_label = pair_dict["t_label"]

    s_mesh = torch.from_numpy(s_mesh_np).unsqueeze(dim=0).to(device)
    t_mesh = torch.from_numpy(t_mesh_np).unsqueeze(dim=0).to(device)
    adv_mesh = torch.from_numpy(adv_mesh_np).unsqueeze(dim=0).to(device)

    with torch.no_grad():
        s_recon_mesh, _ = AE(s_mesh)
        t_recon_mesh, _ = AE(t_mesh)
        adv_recon_mesh, _ = AE(adv_mesh)

    s_recon_mesh_np = s_recon_mesh[0].detach().cpu().numpy()
    t_recon_mesh_np = t_recon_mesh[0].detach().cpu().numpy()
    adv_recon_mesh_np = adv_recon_mesh[0].detach().cpu().numpy()

    if dataset == 'coma':
        vis_numbers = [4784, 4220, 2268, 443, 2188, 1609, 1651, 4034]
    else:  # dataset == 'smal'
        vis_numbers = [374, 828, 458, 852, 260, 513]

    if visualize and pair_number in vis_numbers:
        visualize_mesh_subplots(s_mesh_np, adv_mesh_np, mesh_faces, title_1='src_' + str(pair_number),
                                title_2='adv_' + str(pair_number))
        visualize_mesh_subplots(t_mesh_np, adv_recon_mesh_np, mesh_faces, title_1='target_' + str(pair_number),
                                title_2='adv_recon_' + str(pair_number))

    dict_out = {"step": step, "pair_number": pair_number,
                "s_label": s_label, "t_label": t_label,
                "s_mesh": s_mesh_np, "t_mesh": t_mesh_np,
                "adv_mesh": adv_mesh_np, "adv_recon_mesh": adv_recon_mesh_np,
                "s_recon_mesh": s_recon_mesh_np, "t_recon_mesh": t_recon_mesh_np}
    dict_out_list.append(dict_out)

results_file_out = os.path.join(results_dir, 'mlp_model_transferability.pickle')
print('saving out dictionary, results_file_out: {}'.format(results_file_out))
with open(results_file_out, 'wb') as handle:
    pickle.dump(dict_out_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


