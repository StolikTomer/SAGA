import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle
import hdf5storage
from mesh_operations import get_vert_connectivity


def get_saga_train_idxs(num_shapes):
    outliers = np.asarray([6710, 6792, 6980]) - 1
    singulars = np.asarray([1354, 1804, 2029, 4283, 4306, 4377, 4433, 5543, 5925, 6464,
                            9365, 9575, 9641, 9862, 10210, 10561, 11434, 11778, 11783, 13963,
                            14097, 14762, 15830, 15947, 15948, 15952, 16515, 16624, 16630, 16632,
                            16635, 19971, 20358])
    remeshed = np.asarray([820, 1200, 7190, 11700, 12500, 14270, 15000, 16300, 19180, 20000]) - 1

    # ---------- Split in train and test ----------
    test_subj = [np.int(x) for x in (np.arange(18531, 20465) - 1)]
    idxs_for_train_val = [np.int(x) for x in np.arange(0, num_shapes, 2) if (np.int(x) not in test_subj
                                                                             and np.int(x) not in singulars
                                                                             and np.int(x) not in outliers
                                                                             and np.int(x) not in remeshed)]

    idxs_for_test_full = [x for x in np.arange(0, num_shapes) if
                          x not in idxs_for_train_val and x not in outliers and x not in singulars]
    idxs_for_test = [idxs_for_test_full[x] for x in np.arange(0, len(idxs_for_test_full), 8)]
    idxs_for_val = [idxs_for_train_val[x] for x in np.arange(0, len(idxs_for_train_val), 10)]
    idxs_for_train = [x for x in idxs_for_train_val if x not in idxs_for_val]

    return idxs_for_train, idxs_for_val, idxs_for_test


class SagaDataset(InMemoryDataset):
    def __init__(self, root_dir, saga_data_file, faces_file, type, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pre_transform = pre_transform
        self.saga_data_file = saga_data_file
        self.faces_file = faces_file

        super(SagaDataset, self).__init__(root_dir, transform, pre_transform)

        if type == 'train':
            data_path = self.processed_paths[0]
        elif type == 'val':
            data_path = self.processed_paths[1]
        elif type == 'test':
            data_path = self.processed_paths[2]
        else:
            raise Exception("train, val and test are supported data types")

        norm_path = self.processed_paths[3]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']
        self.data, self.slices = torch.load(data_path)
        if self.transform:
            self.data = [self.transform(td) for td in self.data]

    @property
    def raw_file_names(self):
        return self.saga_data_file

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'val.pt', 'test.pt', 'norm.pt']
        return processed_files

    def process(self):
        all_train_data, all_val_data, all_test_data = [], [], []
        data = hdf5storage.loadmat(self.saga_data_file)  # Load dataset
        mesh_vertices = data['meshes_noeye'].reshape(data['meshes_noeye'].shape[0], data['meshes_noeye'].shape[1],
                                                     3).astype('float32')  # Vertices of the meshes
        num_shapes_in_all_datasets = mesh_vertices.shape[0]

        idxs_for_train, idxs_for_val, idxs_for_test = get_saga_train_idxs(num_shapes_in_all_datasets)
        meshes_test_np = mesh_vertices[idxs_for_test, :, :]
        meshes_val_np = mesh_vertices[idxs_for_val, :, :]
        meshes_train_np = mesh_vertices[idxs_for_train, :, :]

        mesh_faces_np = np.load(self.faces_file)

        for i in range(len(meshes_train_np)):
            train_verts = torch.Tensor(meshes_train_np[i])
            train_adjacency = get_vert_connectivity(meshes_train_np[i], mesh_faces_np).tocoo()
            train_edge_index = torch.Tensor(np.vstack((train_adjacency.row, train_adjacency.col)))
            train_data = Data(x=train_verts, y=train_verts, edge_index=train_edge_index)
            all_train_data.append(train_data)

        for i in range(len(meshes_val_np)):
            val_verts = torch.Tensor(meshes_val_np[i])
            val_adjacency = get_vert_connectivity(meshes_val_np[i], mesh_faces_np).tocoo()
            val_edge_index = torch.Tensor(np.vstack((val_adjacency.row, val_adjacency.col)))
            val_data = Data(x=val_verts, y=val_verts, edge_index=val_edge_index)
            all_val_data.append(val_data)

        for i in range(len(meshes_test_np)):
            test_verts = torch.Tensor(meshes_test_np[i])
            test_adjacency = get_vert_connectivity(meshes_test_np[i], mesh_faces_np).tocoo()
            test_edge_index = torch.Tensor(np.vstack((test_adjacency.row, test_adjacency.col)))
            test_data = Data(x=test_verts, y=test_verts, edge_index=test_edge_index)
            all_test_data.append(test_data)

        mean_train = torch.Tensor(np.mean(meshes_train_np, axis=0))
        std_train = torch.Tensor(np.std(meshes_train_np, axis=0))
        norm_dict = {'mean': mean_train, 'std': std_train}
        if self.pre_transform is not None:
            if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
                if self.pre_transform.mean is None:
                    self.pre_transform.mean = mean_train
                if self.pre_transform.std is None:
                    self.pre_transform.std = std_train
            all_train_data = [self.pre_transform(td) for td in all_train_data]
            all_val_data = [self.pre_transform(td) for td in all_val_data]
            all_test_data = [self.pre_transform(td) for td in all_test_data]

        torch.save(self.collate(all_train_data), self.processed_paths[0])
        torch.save(self.collate(all_val_data), self.processed_paths[1])
        torch.save(self.collate(all_test_data), self.processed_paths[2])
        torch.save(norm_dict, self.processed_paths[3])


class AttackedDataset(InMemoryDataset):
    def __init__(self, root_dir, results_file_in, faces_file, type, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pre_transform = pre_transform
        self.results_file_in = results_file_in
        self.faces_file = faces_file

        super(AttackedDataset, self).__init__(root_dir, transform, pre_transform)

        if type == 'source':
            data_path = self.processed_paths[0]
        elif type == 'target':
            data_path = self.processed_paths[1]
        elif type == 'adversary':
            data_path = self.processed_paths[2]
        else:
            raise Exception("source, target and adversary are supported data types")

        self.data, self.slices = torch.load(data_path)
        if self.transform:
            self.data = [self.transform(td) for td in self.data]

    @property
    def raw_file_names(self):
        return self.results_file_in

    @property
    def processed_file_names(self):
        processed_files = ['source.pt', 'target.pt', 'adversary.pt']
        return processed_files

    def process(self):
        all_source_data, all_target_data, all_adversary_data = [], [], []
        with open(self.results_file_in, 'rb') as handle:
            results_dict = pickle.load(handle)
        faces_np = np.load(self.faces_file)

        pair_numbers = [item["pair_number"] for item in results_dict]
        s_labels = [item["s_label"] for item in results_dict]
        t_labels = [item["t_label"] for item in results_dict]
        s_meshes = [item["s_mesh"] for item in results_dict]
        t_meshes = [item["t_mesh"] for item in results_dict]
        adv_meshes = [item["adv_mesh"] for item in results_dict]

        for i in range(len(s_meshes)):
            s_verts = torch.Tensor(s_meshes[i])
            s_adjacency = get_vert_connectivity(s_meshes[i], faces_np).tocoo()
            s_edge_index = torch.Tensor(np.vstack((s_adjacency.row, s_adjacency.col)))
            s_data = Data(x=s_verts, y=s_verts, edge_index=s_edge_index)
            all_source_data.append(s_data)

            t_verts = torch.Tensor(t_meshes[i])
            t_adjacency = get_vert_connectivity(t_meshes[i], faces_np).tocoo()
            t_edge_index = torch.Tensor(np.vstack((t_adjacency.row, t_adjacency.col)))
            t_data = Data(x=t_verts, y=t_verts, edge_index=t_edge_index)
            all_target_data.append(t_data)

            adv_verts = torch.Tensor(adv_meshes[i])
            adv_adjacency = get_vert_connectivity(adv_meshes[i], faces_np).tocoo()
            adv_edge_index = torch.Tensor(np.vstack((adv_adjacency.row, adv_adjacency.col)))
            adv_data = Data(x=adv_verts, y=adv_verts, edge_index=adv_edge_index)
            all_adversary_data.append(adv_data)

        all_source_data = [self.pre_transform(td) for td in all_source_data]
        all_target_data = [self.pre_transform(td) for td in all_target_data]
        all_adversary_data = [self.pre_transform(td) for td in all_adversary_data]

        torch.save(self.collate(all_source_data), self.processed_paths[0])
        torch.save(self.collate(all_target_data), self.processed_paths[1])
        torch.save(self.collate(all_adversary_data), self.processed_paths[2])
