import os
import torch
import datetime
from utils import log_string, get_device, get_file, visualize_mesh_subplots, get_directories, get_argument_parser
from utils_data import load_data
from utils_models import load_trained_ae, get_ae_params

flags = get_argument_parser()

dataset = flags.dataset
visualize = flags.visualize
show_src_heatmap = flags.show_src_heatmap
purpose = flags.purpose
rand_seed = flags.seed

torch.manual_seed(rand_seed)
ae_params = get_ae_params()
data_dir, models_dir, logs_dir, _, _ = get_directories(dataset=dataset)
autoencoder_dir = os.path.join(models_dir, 'autoencoders')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = get_file(logs_dir, 'evaluate_ae_' + dataset + '_' + current_time + '.txt')
device = get_device(log_file=log_file)
ae_params['purpose'] = purpose
log_string(log_file, 'seed = {}'.format(rand_seed))
log_string(log_file, 'ae params:\n\n{}\n\n'.format(ae_params))

# ---------- Load data ----------
data_struct = load_data(dataset=dataset, params=ae_params, data_dir=data_dir, device=device, log_file=log_file)
num_shapes = data_struct.get_num_shapes()
num_vertices = data_struct.get_num_vertices()
num_dims = data_struct.get_pos(0).shape[-1]

# ---------- Load Model ----------
input_shape = [num_vertices, num_dims]
AE = load_trained_ae(ae_params, autoencoder_dir, device, input_shape)

mean_l2_loss = []
for idx in range(num_shapes):
    mesh = data_struct.get_pos(idx).unsqueeze(dim=0)
    mesh_recon = AE(mesh)[0]

    mesh_np = mesh.detach().cpu().numpy()
    faces_np = data_struct.get_faces().detach().cpu().numpy()
    mesh_recon_np = mesh_recon.detach().cpu().numpy()

    l2_for_x_recon = torch.mean(torch.mean(torch.square(mesh_recon - mesh), dim=-1))
    log_string(log_file, "For purpose: {}, idx: {}, l2_for_x_recon: {}"
               .format(ae_params['purpose'], idx, l2_for_x_recon))
    mean_l2_loss.append(l2_for_x_recon)

    #if data_struct.get_label(idx) == 11:  # use this line for out-of-distribution (ood) evaluations
    if visualize and idx % 500 == 0:
        visualize_mesh_subplots(mesh_np[0], mesh_recon_np[0], faces_np)

log_string(log_file, "mean l2 recon loss: {}".format(torch.mean(torch.tensor(mean_l2_loss))))
