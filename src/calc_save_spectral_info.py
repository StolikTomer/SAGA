import os
import torch
import hdf5storage
import numpy as np
import datetime
import time
from utils import log_string, get_file, get_directories, get_device, get_argument_parser
from utils_data import get_data_indices, find_shared_evects_basis
from spectral_torch import calculate_eigenvalues_batch
from smal_dataset import SmalDataset

debug_mode = False

flags = get_argument_parser()

rand_seed = flags.seed
torch.manual_seed(rand_seed)
if debug_mode:
    dataset = 'coma'
    num_of_eigenvalues = 30
    num_of_eigenvectors = 30
    check_saving = False
    image_suffix = '.png'
else:
    dataset = flags.dataset
    num_of_eigenvalues = flags.num_evals
    num_of_eigenvectors = 3000
    check_saving = flags.check_saving
    image_suffix = flags.image_suffix

# ---------- Configurations ----------
purpose = 'test'
reduced_memory_mode = 'weak'
data_dir, _, logs_dir, _, images_dir = get_directories(dataset=dataset)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = get_file(logs_dir, 'save_calc_eigenvalues_' + dataset + '_save_' + str(num_of_eigenvalues) + '_at_' + current_time + '.txt')
log_string(log_file, "seed = {}".format(rand_seed))
log_string(log_file, "number of eigenvalues = {}".format(num_of_eigenvalues))
log_string(log_file, "number of eigenvectors = {}".format(num_of_eigenvectors))
log_string(log_file, "check saving = {}".format(check_saving))
log_string(log_file, "purpose = {}".format(purpose))
log_string(log_file, "reduced_memory_mode = {}".format(reduced_memory_mode))
device = get_device(log_file=log_file)
#device = 'cpu'
visualize = False
save_images = False
save_shared_basis_train_image = os.path.join(images_dir, 'shared_basis_train' + image_suffix)

# Load data
log_string(log_file, 'loading ' + dataset + ' data ...')
if dataset == 'coma':
    raw_data_file = os.path.join(data_dir, 'raw', 'coma_FEM.mat')
    assert os.path.exists(raw_data_file), 'coma raw data file does not exist'
    data = hdf5storage.loadmat(raw_data_file)  # Load dataset
    mesh_vertices = data['meshes_noeye'].reshape(data['meshes_noeye'].shape[0], data['meshes_noeye'].shape[1], 3).astype('float32')  # Vertices of the meshes
    mesh_faces = data['f_noeye'] - 1
    e_data = data['noeye_evals_FEM3'][:, 0:num_of_eigenvalues].astype('float32')  # Eigenvalues of the meshes
    idxs_for_train, idxs_for_val, idxs_for_test = get_data_indices(dataset, reduced_memory_mode, num_shapes=mesh_vertices.shape[0])

    mesh_vertices = torch.from_numpy(mesh_vertices.astype(np.float32)).to(device)
    saved_faces = torch.from_numpy(mesh_faces.astype(np.int64)).to(device)

    if purpose == 'train':
        idxs_to_save = idxs_for_train
    elif purpose == 'val':
        idxs_to_save = idxs_for_val
    else:  # purpose == 'test'
        idxs_to_save = idxs_for_test

    saved_meshes = mesh_vertices[idxs_to_save, :, :]
    num_shapes = len(saved_meshes)
else:  # dataset == 'smal'
    customdata = SmalDataset(data_dir, device=device, train=False, test=False, custom=False, transform_data=True)
    idxs_for_train, idxs_for_val, idxs_for_test = get_data_indices(dataset, reduced_memory_mode, customdata=customdata)
    if purpose == "train":
        customdata = customdata[idxs_for_train]
    elif purpose == "val":
        customdata = customdata[idxs_for_val]
    else:  # purpose == "test"
        customdata = customdata[idxs_for_test]

    num_shapes = len(customdata)
    num_vertices = customdata[0].pos.shape[0]
    saved_meshes = torch.zeros(size=(num_shapes, num_vertices, 3), dtype=torch.float32)
    saved_faces = customdata[0].face.t()

    for idx in range(num_shapes):
        saved_meshes[idx, :, :] = customdata[idx].pos

shared_basis_jump = 40
idxs_for_shared_basis = [np.int(x) for x in np.arange(0, num_shapes, shared_basis_jump)]
shared_meshes = saved_meshes[idxs_for_shared_basis]
log_string(log_file, 'Starting eigenvalues and eigenvectors calculations...')
e_calc, v_calc = \
    calculate_eigenvalues_batch(shared_meshes, saved_faces, num_of_eigenvalues, device, num_of_eigenvectors,
                                return_vectors=True, log_file=log_file)
assert len(e_calc) == len(shared_meshes)  # if we fail on some shape we will get mixed-up indices - hard to debug

log_string(log_file, 'Starting shared basis calculations...')
v_shared, basis_coeff = find_shared_evects_basis(shared_meshes, v_calc, device,
                                                 visualize=visualize, save_images=save_images,
                                                 save_shared_basis_train_image=save_shared_basis_train_image,
                                                 log_file=log_file)

a_save = np.zeros((saved_meshes.shape[0], v_calc.shape[2], 3))
log_string(log_file, 'Starting alphas calculations...')
for idx in range(len(saved_meshes)):
    start_alpha_calc_time = time.time()
    v_i = saved_meshes[idx].to(device)
    alphas_i = torch.linalg.lstsq(v_shared, v_i)[0]
    a_save[idx, :, :] = alphas_i.detach().cpu().numpy()
    log_string(log_file, 'spectral coefficients calculation time for a single shape (seconds): {}'
               .format(time.time() - start_alpha_calc_time))

v_shared_save = v_shared.detach().cpu().numpy()
basis_coeff_save = basis_coeff.detach().cpu().numpy()

log_string(log_file, "v_shared_save.shape={}, a_save.shape={}"
           .format(v_shared_save.shape, a_save.shape))
log_string(log_file, "basis_coeff=\n{}".format(basis_coeff_save))

spectral_data = os.path.join(data_dir, 'spectral')
if not os.path.exists(spectral_data):
    os.mkdir(spectral_data)
save_file_evects_shared = os.path.join(spectral_data, 'shared_basis_' + str(num_of_eigenvectors) + '_' + purpose + '.npy')
save_file_alphas = os.path.join(spectral_data, 'alphas_' + str(num_of_eigenvectors) + '_' + purpose + '.npy')

np.save(save_file_evects_shared, v_shared_save)
np.save(save_file_alphas, a_save)

if check_saving:
    log_string(log_file, "checking saved files...")
    check_shared_evects = np.load(save_file_evects_shared)
    check_alphas = np.load(save_file_alphas)

    log_string(log_file, "check_shared_evects.shape: {}".format(check_shared_evects.shape))
    log_string(log_file, "check_alphas.shape: {}".format(check_alphas.shape))
