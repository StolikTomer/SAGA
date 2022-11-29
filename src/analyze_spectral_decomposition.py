import torch
import datetime
import matplotlib.pylab as plt
from utils import log_string, get_device, get_file, visualize_mesh_subplots, get_directories, get_argument_parser
from utils_data import load_data
from utils_attack import get_attack_params
from plotly_visualization import visualize_and_compare
from spectral_torch import calculate_eigenvalues_batch

flags = get_argument_parser()
show_self_basis = False

dataset = flags.dataset
num_shared_evects = flags.num_shared_evects
reduced_memory_mode = flags.reduced_memory_mode
visualize = flags.visualize
show_src_heatmap = flags.show_src_heatmap
rand_seed = flags.seed
torch.manual_seed(rand_seed)
params = get_attack_params(dataset=dataset)
params['N_evects'] = num_shared_evects
params['reduced_memory_mode'] = reduced_memory_mode

data_dir, _, logs_dir, _, _ = get_directories(dataset=dataset)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = get_file(logs_dir, 'analyze_spectral_decomposition_' + dataset + '_' + current_time + '.txt')
log_string(log_file, 'seed = {}'.format(rand_seed))
log_string(log_file, 'params:\n\n{}\n\n'.format(params))
device = get_device(log_file=log_file)
analyze_list = [18, 32, 66, 118, 156] if dataset == 'coma' else [2, 34, 55, 62, 92]

# ---------- Load data ----------
data_struct = load_data(dataset=dataset, params=params, data_dir=data_dir, device=device, log_file=log_file)
evects_shared = data_struct.get_evects()
evects_shared_np = evects_shared.detach().cpu().numpy()

log_string(log_file, 'number of shapes: {}'.format(data_struct.get_num_shapes()))
log_string(log_file, 'start data analysis ...')

recon_error_list = []
recon_error_from_shared_list = []
for idx in range(data_struct.get_num_shapes()):
    # prepare data for attack
    M = data_struct.get_pos(idx)

    alphas_from_shared = torch.linalg.lstsq(evects_shared, M)[0]
    M_recon_from_shared = torch.matmul(evects_shared, alphas_from_shared)
    recon_error_from_shared = torch.mean((M_recon_from_shared - M).pow(2))
    recon_error_from_shared_list.append(recon_error_from_shared)

    if show_self_basis:
        M_unsqueezed = M.unsqueeze(dim=0)
        faces = data_struct.get_faces()
        evects_num = (3931 - 2) if dataset == 'coma' else (3889 - 2)
        _, M_evects = calculate_eigenvalues_batch(M_unsqueezed, faces, 30, device, evects_num,
                                                  log_file=log_file, return_vectors=True)

        evects = M_evects[0]
        alphas = torch.linalg.lstsq(evects, M)[0]
        M_recon = torch.matmul(evects, alphas)
        recon_error = torch.mean((M_recon - M).pow(2))
        recon_error_list.append(recon_error)

        log_string(log_file, "For shape %d, recon_error: %.3e, recon_error_from_shared: %.3e" % (
            idx, recon_error, recon_error_from_shared))
    else:
        log_string(log_file, "For shape %d, recon_error_from_shared: %.3e" % (idx, recon_error_from_shared))

    if visualize and idx in analyze_list:
        mesh_faces_np = data_struct.get_faces().detach().cpu().numpy()
        M_v = M.detach().cpu().numpy()
        M_recon_from_shared_v = M_recon_from_shared.detach().cpu().numpy()
        visualize_mesh_subplots(M_v, M_recon_from_shared_v, mesh_faces_np, title_1='Mesh',
                                title_2='Recon from shared eigenvectors')
        if show_self_basis:
            M_recon_v = M_recon.detach().cpu().numpy()
            visualize_mesh_subplots(M_v, M_recon_v, mesh_faces_np, title_1='Mesh',
                                    title_2='Recon from self eigenvectors')


        recon_error_tmp = []
        recon_error_std_tmp = []
        for n in range(100, evects_shared.shape[1], 100):
            phi_tmp = evects_shared[:, :n]
            alpha_tmp = alphas_from_shared[:n, :]
            M_recon_tmp = torch.matmul(phi_tmp, alpha_tmp)
            recon_error_tmp.append(torch.mean((M_recon_tmp - M).pow(2)))
            recon_error_std_tmp.append(torch.std((M_recon_tmp - M).pow(2)))

            if n in [100, 500, 1000, 2000]:
                M_recon_tmp_v = M_recon_tmp.detach().cpu().numpy()
                visualize_mesh_subplots(M_v, M_recon_tmp_v, mesh_faces_np,
                                        title_1='Mesh', title_2='Recon from shared ' + str(n) + ' eigenvectors')
                if show_src_heatmap:
                    visualize_and_compare(M_recon_tmp, data_struct.get_faces(), M, data_struct.get_faces(),
                                          (M - M_recon_tmp).norm(p=2, dim=-1))

        x_axis = [i for i in range(100, evects_shared.shape[1], 100)]
        plt.figure('Mean Spectral Reconstruction vs. Number of Eigenvectors')
        plt.title('Mean Spectral Reconstruction vs. Number of Eigenvectors')
        plt.plot(x_axis, recon_error_tmp, 'red', label='mean')
        plt.legend()
        plt.xlabel('Number of Eigenvectors')
        plt.ylabel('Spectral Reconstruction L2')
        plt.show()

        plt.figure('STD Spectral Reconstruction vs. Number of Eigenvectors')
        plt.title('STD Spectral Reconstruction vs. Number of Eigenvectors')
        plt.plot(x_axis, recon_error_std_tmp, 'blue', label='std')
        plt.legend()
        plt.xlabel('Number of Eigenvectors')
        plt.ylabel('Spectral Reconstruction STD')
        plt.show()

if show_self_basis:
    log_string(log_file, "average recon_error: {}, average recon_error_from_shared: {}"
               .format(torch.mean(torch.tensor(recon_error_list)), torch.mean(torch.tensor(recon_error_from_shared_list))))
else:
    log_string(log_file, "average recon_error_from_shared: {}".format(torch.mean(torch.tensor(recon_error_from_shared_list))))
