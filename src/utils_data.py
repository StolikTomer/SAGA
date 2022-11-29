import os
import numpy as np
import torch
import time
from utils import log_string
import hdf5storage
from smal_dataset import SmalDataset
import matplotlib.pylab as plt


def get_data_indices(dataset, reduced_memory_mode, num_shapes=0, customdata=None):
    if dataset == 'coma':
        outliers = np.asarray([6710, 6792, 6980]) - 1
        singulars = np.asarray([1354, 1804, 2029, 4283, 4306, 4377, 4433, 5543, 5925, 6464,
                                9365, 9575, 9641, 9862, 10210, 10561, 11434, 11778, 11783, 13963,
                                14097, 14762, 15830, 15947, 15948, 15952, 16515, 16624, 16630, 16632,
                                16635, 19971, 20358])
        remeshed = np.asarray([820, 1200, 7190, 11700, 12500, 14270, 15000, 16300, 19180, 20000]) - 1

        # ---------- Split in train and test ----------
        ood_identity = [np.int(x) for x in (np.arange(18531, 20465) - 1)]
        idxs_for_train_val = [np.int(x) for x in np.arange(0, num_shapes, 2) if (np.int(x) not in ood_identity
                                                                                 and np.int(x) not in singulars
                                                                                 and np.int(x) not in outliers
                                                                                 and np.int(x) not in remeshed)]
        idxs_for_test_full = [x for x in np.arange(0, num_shapes) if
                              x not in idxs_for_train_val and x not in outliers and x not in singulars]

        idxs_for_val_full = [idxs_for_train_val[x] for x in np.arange(0, len(idxs_for_train_val), 10)]
        idxs_for_train_full = [x for x in idxs_for_train_val if x not in idxs_for_val_full]

        if reduced_memory_mode == 'strong':
            idxs_for_test = [idxs_for_test_full[x] for x in np.arange(0, len(idxs_for_test_full), 100)]

            idxs_for_val = idxs_for_val_full

            idxs_for_train = [idxs_for_train_full[x] for x in np.arange(0, len(idxs_for_train_full), 3)]
        elif reduced_memory_mode == 'weak':
            idxs_for_test = [idxs_for_test_full[x] for x in np.arange(0, len(idxs_for_test_full), 8)]

            idxs_for_val = idxs_for_val_full

            idxs_for_train = [idxs_for_train_full[x] for x in np.arange(0, len(idxs_for_train_full), 3)]
        else:  # reduced_memory_mode == 'none'
            idxs_for_test = idxs_for_test_full

            idxs_for_val = idxs_for_val_full

            idxs_for_train = idxs_for_train_full

    else:  # dataset == 'smal'
        assert customdata is not None, "get_data_indices: customdata parameter is requires for SMAL dataset"
        num_shapes = len(customdata)
        singulars = np.asarray([1929, 2083, 2115, 2120, 2696, 2894, 3524, 5824, 7310, 9135])
        idxs_for_train = [np.int(x) for x in np.arange(0, num_shapes) if ((customdata[x].model == 2)
                                                                          and np.int(x) not in singulars)]
        idxs_for_val = [np.int(x) for x in np.arange(0, num_shapes) if ((customdata[x].model == 1)
                                                                        and np.int(x) not in singulars)]
        idxs_for_test = [np.int(x) for x in np.arange(0, num_shapes) if ((customdata[x].model == 0)
                                                                         and np.int(x) not in singulars)]

    return idxs_for_train, idxs_for_val, idxs_for_test


def build_label_vector_coma(slice_idxs, num_shapes):
    shape_labels = np.zeros(shape=[num_shapes])
    for i in range(num_shapes):
        for j in range(len(slice_idxs)):
            if i < slice_idxs[j]:
                shape_labels[i] = j - 1
                break
    return shape_labels


def load_data(dataset, params, data_dir, device, log_file=None):
    log_string(log_file, 'loading ' + dataset + ' data ...')

    purpose = params['purpose']
    reduced_memory_mode = params['reduced_memory_mode']
    num_evects_in_file = 3000
    log_string(log_file, 'purpose={}, num_eigenvectors_in_file={}, reduced_memory_mode={}'
               .format(purpose, num_evects_in_file, reduced_memory_mode))

    spectral_data = os.path.join(data_dir, 'spectral')
    shared_evects_path = os.path.join(spectral_data, 'shared_basis_' + str(
                                      num_evects_in_file) + '_' + purpose + '.npy')
    alphas_path = os.path.join(spectral_data, 'alphas_' + str(
                               num_evects_in_file) + '_' + purpose + '.npy')

    if dataset == 'coma':
        raw_data_file = os.path.join(data_dir, 'raw', 'coma_FEM.mat')
        assert os.path.exists(raw_data_file), 'coma raw data file does not exist'
        data = hdf5storage.loadmat(raw_data_file)  # Load dataset
        mesh_vertices = data['meshes_noeye'].reshape(data['meshes_noeye'].shape[0], data['meshes_noeye'].shape[1],
                                                     3).astype('float32')  # Vertices of the meshes
        mesh_faces_np = data['f_noeye'] - 1
        # e_data = data['noeye_evals_FEM3'].astype('float32')  # Eigenvalues of the meshes
        num_shapes_in_all_datasets = mesh_vertices.shape[0]

        idxs_for_train, idxs_for_val, idxs_for_test = get_data_indices(dataset=dataset,
                                                                       reduced_memory_mode=reduced_memory_mode,
                                                                       num_shapes=num_shapes_in_all_datasets)

        if purpose == 'test':
            vertices_np = mesh_vertices[idxs_for_test, :, :]
        elif purpose == 'val':
            vertices_np = mesh_vertices[idxs_for_val, :, :]
        else:  # purpose == 'train'
            vertices_np = mesh_vertices[idxs_for_train, :, :]

        num_shapes_in_dataset = len(vertices_np)

        log_string(log_file, 'adding labels...')
        labels_np = build_label_vector_coma(get_slice_idx(dataset=dataset, params=params), num_shapes_in_dataset)

        vertices = torch.from_numpy(vertices_np.astype(np.float32)).to(device)
        faces = torch.from_numpy(mesh_faces_np.astype(np.int)).to(device=device)
        labels = torch.from_numpy(labels_np.astype(np.int)).to(device=device)
    else:  # dataset == 'smal'
        customdata = SmalDataset(data_dir, device=device, train=False, test=False, custom=False, transform_data=True)
        idxs_for_train, idxs_for_val, idxs_for_test = get_data_indices(dataset=dataset,
                                                                       reduced_memory_mode=reduced_memory_mode,
                                                                       customdata=customdata)
        if purpose == "train":
            data = customdata[idxs_for_train]
        elif purpose == "val":
            data = customdata[idxs_for_val]
        if purpose == "test":
            data = customdata[idxs_for_test]

        vertices = torch.stack([data[x].pos for x in range(len(data))]).to(device=device, dtype=torch.float32)
        labels = torch.stack([data[x].y for x in range(len(data))]).to(device=device, dtype=torch.int64)
        faces = data[0].face.t().to(device=device, dtype=torch.int64)

    log_string(log_file, 'loading shared eigenvectors basis...')
    if os.path.exists(shared_evects_path):
        shared_evects_np = np.load(shared_evects_path)
    else:
        log_string(log_file, 'shared eigenvectors file does not exist, using a dummy struct...')
        shared_evects_np = np.zeros((vertices.shape[1], num_evects_in_file))

    log_string(log_file, 'loading alphas...')
    if os.path.exists(alphas_path):
        alphas_np = np.load(alphas_path)
    else:
        log_string(log_file, 'alphas file does not exist, using a dummy struct...')
        alphas_np = np.zeros((vertices.shape[0], num_evects_in_file, 3))

    shared_evects = torch.from_numpy(shared_evects_np.astype(np.float32)).to(device=device)
    alphas = torch.from_numpy(alphas_np.astype(np.float32)).to(device=device)

    shared_evects = shared_evects[:, 0:num_evects_in_file]
    alphas = alphas[:, 0:num_evects_in_file, :]

    return Data(vertices, alphas, labels, faces, shared_evects, device)


def find_shared_evects_basis(shared_meshes, shared_evects, device, visualize=False,
                             save_images=False, save_shared_basis_train_image=None, log_file=None):
    num_shapes = len(shared_meshes)
    base_optimizer = SharedEigenBasis(num_shapes, device, dtype=shared_evects.dtype)
    base_optimizer.init_parameters()
    shared_basis = base_optimizer.optimize(shared_meshes, shared_evects,
                                           visualize=visualize, save_images=save_images,
                                           save_shared_basis_train_image=save_shared_basis_train_image,
                                           log_file=log_file)
    return shared_basis, base_optimizer.get_basis_coeff()


def get_slice_idx(dataset, params):
    purpose = params['purpose']
    reduced_memory_mode = params['reduced_memory_mode']
    if dataset == 'coma':
        if purpose == 'test':
            if reduced_memory_mode == 'strong':
                slice_idx_0 = 0
                slice_idx_1 = 6
                slice_idx_2 = 14
                slice_idx_3 = 22
                slice_idx_4 = 33
                slice_idx_5 = 44
                slice_idx_6 = 50
                slice_idx_7 = 59
                slice_idx_8 = 67
                slice_idx_9 = 77
                slice_idx_10 = 84
                slice_idx_11 = 93
                slice_idx_last = 112
            elif reduced_memory_mode == 'weak':
                slice_idx_0 = 0
                slice_idx_1 = 74
                slice_idx_2 = 174
                slice_idx_3 = 265
                slice_idx_4 = 404
                slice_idx_5 = 540
                slice_idx_6 = 620
                slice_idx_7 = 735
                slice_idx_8 = 838
                slice_idx_9 = 953
                slice_idx_10 = 1049
                slice_idx_11 = 1156
                slice_idx_last = 1398
            else:  # reduced_memory_mode == 'none'
                slice_idx_0 = 0
                slice_idx_1 = 591
                slice_idx_2 = 1387
                slice_idx_3 = 2116
                slice_idx_4 = 3231
                slice_idx_5 = 4314
                slice_idx_6 = 4955
                slice_idx_7 = 5878
                slice_idx_8 = 6699
                slice_idx_9 = 7621
                slice_idx_10 = 8392
                slice_idx_11 = 9246
                slice_idx_last = 11178
        elif purpose == 'val':
            slice_idx_0 = 0
            slice_idx_1 = 60
            slice_idx_2 = 139
            slice_idx_3 = 212
            slice_idx_4 = 324
            slice_idx_5 = 432
            slice_idx_6 = 497
            slice_idx_7 = 589
            slice_idx_8 = 671
            slice_idx_9 = 763
            slice_idx_10 = 840
            slice_idx_last = 926
        else:  # purpose == 'train'
            if reduced_memory_mode == 'strong' or reduced_memory_mode == 'weak':
                slice_idx_0 = 0
                slice_idx_1 = 178
                slice_idx_2 = 416
                slice_idx_3 = 635
                slice_idx_4 = 970
                slice_idx_5 = 1296
                slice_idx_6 = 1489
                slice_idx_7 = 1766
                slice_idx_8 = 2012
                slice_idx_9 = 2289
                slice_idx_10 = 2519
                slice_idx_last = 2775
            else:  # reduced_memory_mode == 'none'
                slice_idx_0 = 0
                slice_idx_1 = 532
                slice_idx_2 = 1247
                slice_idx_3 = 1903
                slice_idx_4 = 2910
                slice_idx_5 = 3888
                slice_idx_6 = 4465
                slice_idx_7 = 5296
                slice_idx_8 = 6034
                slice_idx_9 = 6866
                slice_idx_10 = 7556
                slice_idx_last = 8325
        if purpose == 'test':
            return [slice_idx_0, slice_idx_1, slice_idx_2, slice_idx_3, slice_idx_4, slice_idx_5, slice_idx_6,
                    slice_idx_7,
                    slice_idx_8, slice_idx_9, slice_idx_10, slice_idx_11, slice_idx_last]
        else:
            return [slice_idx_0, slice_idx_1, slice_idx_2, slice_idx_3, slice_idx_4, slice_idx_5, slice_idx_6,
                    slice_idx_7,
                    slice_idx_8, slice_idx_9, slice_idx_10, slice_idx_last]
    else:  # dataset == 'smal'
        if purpose == 'train':
            slice_idx_0 = 0
            slice_idx_1 = 1689
            slice_idx_2 = 3381
            slice_idx_3 = 5026
            slice_idx_4 = 6724
            slice_idx_last = 8430
        elif purpose == 'val':
            slice_idx_0 = 0
            slice_idx_1 = 207
            slice_idx_2 = 405
            slice_idx_3 = 598
            slice_idx_4 = 799
            slice_idx_last = 991
        else:  # purpose == 'test'
            slice_idx_0 = 0
            slice_idx_1 = 83
            slice_idx_2 = 187
            slice_idx_3 = 296
            slice_idx_4 = 396
            slice_idx_last = 497
        return [slice_idx_0, slice_idx_1, slice_idx_2, slice_idx_3, slice_idx_4, slice_idx_last]


def convert_class_name_to_label(classname, dataset, device):
    if dataset == 'coma':
        label = torch.tensor(int(classname.split('_')[1]) - 1)
    else:  # dataset == 'smal'
        label = torch.tensor(0).to(device) if classname == 'cat' else \
            torch.tensor(1).to(device) if classname == 'cow' else \
                torch.tensor(2).to(device) if classname == 'dog' else \
                    torch.tensor(3).to(device) if classname == 'hippo' else \
                        torch.tensor(4).to(device)  # classname is 'horse'
    return label


class SharedEigenBasis:
    def __init__(self, num_shapes, device, dtype=torch.float, lr=1e-5):
        self.device = device
        self.lr = lr
        self.dtype = dtype
        self.num_shapes = num_shapes
        self.basis_coeff = None
        self.optimizer = None

    def init_parameters(self):
        self.basis_coeff = torch.zeros([self.num_shapes],
                                       device=self.device,
                                       dtype=self.dtype,
                                       requires_grad=True)
        with torch.no_grad():
            self.basis_coeff[0] = 1
        self.optimizer = torch.optim.Adam([self.basis_coeff], lr=self.lr)

    def diff_lstq(self, A, Y, lamb=0.010, log_file=None):
        """
        Differentiable least square
        param A: m x n
        param Y: n x 1
        """
        # Assuming A to be full column rank
        cols = A.shape[1]
        tmp_start = time.time()
        if cols == torch.linalg.matrix_rank(A):
            q, r = torch.linalg.qr(A)
            x = torch.inverse(r) @ q.T @ Y
        else:
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols)
            Y_dash = A.permute(1, 0) @ Y
            x = self.diff_lstq(A_dash, Y_dash)
        log_string(log_file, 'time of a single lstsq calc (seconds): {}'.format(time.time() - tmp_start))
        return x

    def optimize(self, meshes, bases, optim_steps=50, visualize=False, save_images=False,
                 save_shared_basis_train_image=None, log_file=None):
        meshes = meshes.to(self.device)
        bases = bases.to(self.device)
        loss_list = []
        mse = torch.nn.MSELoss()
        for step in range(optim_steps):
            shared_basis = torch.tensordot(self.basis_coeff, bases, dims=([0], [0]))

            start_step_time = time.time()
            loss = 0
            for idx in range(meshes.shape[0]):
                mesh = meshes[idx]
                # alphas_i = shared_basis.pinverse() @ mesh
                # alphas_i = torch.linalg.lstsq(shared_basis, mesh)[0]
                alphas_i = self.diff_lstq(shared_basis, mesh, log_file=log_file)

                mesh_recon = torch.matmul(shared_basis, alphas_i)

                loss += mse(mesh, mesh_recon)

            loss_list.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            end_step_time = time.time()
            step_duration = (end_step_time - start_step_time) / 60
            template = 'shared basis optimization step {} out of {}, duration (minutes) {}, loss: {}'
            log_string(log_file, 'basis coefficients: {}'.format(self.basis_coeff))
            log_string(log_file, template.format(step + 1, optim_steps,
                                                 step_duration, loss))

        if visualize:
            plt.figure('Shared Basis Optimization Steps Loss')
            plt.title('Shared Basis Optimization Steps Loss')
            plt.plot(loss_list)
            plt.xlabel('Optimization Step')
            plt.ylabel('Loss Value')
            if save_images and save_shared_basis_train_image is not None:
                plt.savefig(save_shared_basis_train_image, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        return shared_basis

    def get_basis_coeff(self):
        return self.basis_coeff


class Data(torch.utils.data.Dataset):
    def __init__(self, pos, alphas, labels, faces, evects, device):
        self.pos = pos
        self.alphas = alphas
        self.labels = labels
        self.faces = faces
        self.evects = evects
        self.device = device

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, index):
        mesh_vertices = self.pos[index]
        eigen_alphas = self.alphas[index]
        mesh_label = self.labels[index]
        return mesh_vertices, eigen_alphas, mesh_label

    def get_pos(self, shape_index) -> torch.Tensor:
        return self.pos[shape_index].to(self.device)

    def get_alphas(self, shape_index) -> torch.Tensor:
        assert self.alphas.sum() != 0, "Data class: alphas are not valid"
        return self.alphas[shape_index].to(self.device)

    def get_label(self, shape_index) -> torch.Tensor:
        return self.labels[shape_index].to(self.device)

    def get_faces(self) -> torch.Tensor:
        return self.faces.to(self.device)

    def get_evects(self) -> torch.Tensor:
        assert self.evects.sum() != 0, "Data class: evects are not valid"
        return self.evects.to(self.device)

    def get_np_poses(self):
        poses = torch.stack([self.get_pos(x) for x in range(self.get_num_shapes())]).to(self.device)
        return poses.detach().cpu().numpy()

    def get_np_alphas(self):
        alphas = torch.stack([self.get_alphas(x) for x in range(self.get_num_shapes())]).to(self.device)
        return alphas.detach().cpu().numpy()

    def get_np_labels(self):
        labels = torch.stack([self.get_label(x) for x in range(self.get_num_shapes())]).to(self.device)
        return labels.detach().cpu().numpy()

    def get_num_shapes(self):
        return self.__len__()

    def get_num_vertices(self):
        return self.pos[0].shape[0]
