import os
import tqdm
import torch_geometric.data
import torch_geometric.transforms as transforms
import torch_geometric.io as gio
import numpy as np
import random
import torch
from typing import List
import math
import heapq
import scipy.sparse as sp


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def vertex_quadrics(mesh_v: np.ndarray, mesh_f: np.ndarray):
    """Computes a quadric for each vertex in the Mesh.
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((len(mesh_v), 4, 4,))

    # For each face...
    for f_idx in range(len(mesh_f)):

        # Compute normalized plane equation for that face
        vert_idxs = mesh_f[f_idx]

        verts = np.hstack((mesh_v[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh_f[f_idx, k], :, :] += np.outer(eq, eq)
    return v_quadrics


def get_vertices_per_edge(mesh_v, mesh_f):
    vc = sp.coo_matrix(get_adjacency_matrix(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:, 0] < result[:, 1]]  # for uniqueness
    return result


def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left), num_original_verts))

    return new_faces, mtx


def qslim_decimator_transformer(mesh_v, mesh_f, factor=None, n_verts_desired=None):
    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')
    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh_v) * factor)

    # compute the quadrics of the mesh
    Qv = vertex_quadrics(mesh_v, mesh_f)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    vert_adj = get_vertices_per_edge(mesh_v, mesh_f)
    data = vert_adj[:, 0] * 0 + 1
    vert_adj = sp.csc_matrix(
        (data, (vert_adj[:, 0], vert_adj[:, 1])),
        shape=(len(mesh_v), len(mesh_v)))

    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh_v)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    # decimate the mesh
    collapse_list = []
    nverts_total = len(mesh_v)
    faces = mesh_f.copy()

    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh_v)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))

    new_faces, mtx = _get_sparse_transform(faces, len(mesh_v))
    return new_faces, mtx


def get_adjacency_matrix(mesh_v: np.ndarray, mesh_f: np.ndarray):
    vpv = sp.csc_matrix((len(mesh_v), len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:, i]
        JS = mesh_f[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T
    return


def generate_transform_matrices(mesh_v: np.ndarray, mesh_f: np.ndarray, factors: List[float]):
    if len(mesh_v.shape) != 2 and mesh_v.shape[1] != 3 and isinstance(mesh_v.dtype, np.floating):
        raise ValueError("input vertex positions must have shape [N,3] and floating point data type")

    if len(mesh_f.shape) != 2 and mesh_f.shape[1] != 3 and isinstance(mesh_v.dtype, np.integer):
        raise ValueError("input vertex positions must have shape [M,3] and integer data type")

    factors = map(lambda x: 1.0 / x, factors)
    V, F, A, D = [], [], [], []
    A.append(get_adjacency_matrix(mesh_v, mesh_f).tocoo())
    V.append(mesh_v)
    F.append(mesh_f)

    for i, factor in enumerate(factors):
        # compute the decimation quadrics
        new_mesh_f, ds_D = qslim_decimator_transformer(V[-1], F[-1], factor=factor)
        D.append(ds_D.tocoo())
        new_mesh_v = ds_D.dot(V[-1])

        pos = torch.from_numpy(new_mesh_v)
        face = torch.from_numpy(new_mesh_f).t()

        V.append(new_mesh_v)
        F.append(new_mesh_f)
        A.append(get_adjacency_matrix(new_mesh_v, new_mesh_f).tocoo())
    return V, F, A, D


class Rotate(object):
    def __init__(self, dims=[0, 1, 2]):
        super().__init__()
        self.dims = dims

    def __call__(self, mesh: torch_geometric.data):
        dims = self.dims
        phi_n = [random.random() * 2 * math.pi for _ in dims]
        cos_n = [math.cos(phi) for phi in phi_n]
        sin_n = [math.sin(phi) for phi in phi_n]

        pos = mesh.pos
        device = pos.device
        R = torch.tensor(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]], device=device, dtype=torch.float)

        random.shuffle(dims)  # add randomness
        for i in range(len(dims)):
            dim, phi = dims[i], phi_n[i]
            cos, sin = math.cos(phi), math.sin(phi)

            if dim == 0:
                tmp = torch.tensor(
                    [[1, 0, 0],
                     [0, cos, -sin],
                     [0, sin, cos]], device=device)
            elif dim == 1:
                tmp = torch.tensor(
                    [[cos, 0, -sin],
                     [0, 1, 0],
                     [sin, 0, cos]], device=device)
            elif dim == 2:
                tmp = torch.tensor(
                    [[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]], device=device)
            R = R.mm(tmp)
        mesh.pos = torch.matmul(pos, R.t())
        return mesh


class Rotate180(object):
    def __init__(self, dims=[0]):
        super().__init__()
        self.dims = dims

    def __call__(self, mesh: torch_geometric.data):
        dims = self.dims
        phi_n = [math.pi for _ in dims]
        cos_n = [math.cos(phi) for phi in phi_n]
        sin_n = [math.sin(phi) for phi in phi_n]

        pos = mesh.pos
        device = pos.device
        R = torch.tensor(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]], device=device, dtype=torch.float)

        random.shuffle(dims)  # add randomness
        for i in range(len(dims)):
            dim, phi = dims[i], phi_n[i]
            cos, sin = math.cos(phi), math.sin(phi)

            if dim == 0:
                tmp = torch.tensor(
                    [[1, 0, 0],
                     [0, cos, -sin],
                     [0, sin, cos]], device=device)
            elif dim == 1:
                tmp = torch.tensor(
                    [[cos, 0, -sin],
                     [0, 1, 0],
                     [sin, 0, cos]], device=device)
            elif dim == 2:
                tmp = torch.tensor(
                    [[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]], device=device)
            R = R.mm(tmp)
        mesh.pos = torch.matmul(pos, R.t())
        return mesh


class Rotate90(object):
    def __init__(self, dims=[0]):
        super().__init__()
        self.dims = dims

    def __call__(self, mesh: torch_geometric.data):
        dims = self.dims
        phi_n = [math.pi / 2 for _ in dims]
        cos_n = [math.cos(phi) for phi in phi_n]
        sin_n = [math.sin(phi) for phi in phi_n]

        pos = mesh.pos
        device = pos.device
        R = torch.tensor(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]], device=device, dtype=torch.float)

        random.shuffle(dims)  # add randomness
        for i in range(len(dims)):
            dim, phi = dims[i], phi_n[i]
            cos, sin = math.cos(phi), math.sin(phi)

            if dim == 0:
                tmp = torch.tensor(
                    [[1, 0, 0],
                     [0, cos, -sin],
                     [0, sin, cos]], device=device)
            elif dim == 1:
                tmp = torch.tensor(
                    [[cos, 0, -sin],
                     [0, 1, 0],
                     [sin, 0, cos]], device=device)
            elif dim == 2:
                tmp = torch.tensor(
                    [[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]], device=device)
            R = R.mm(tmp)
        mesh.pos = torch.matmul(pos, R.t())
        return mesh


class Move(object):
    def __init__(self, mean=[0, 0, 0], std=[0.05, 0.05, 0.05]):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, mesh: torch_geometric.data):
        pos = mesh.pos
        n = pos.shape[0]
        comp_device = pos.device
        comp_type = pos.dtype
        mean = torch.tensor([self.mean], device=comp_device, dtype=comp_type)
        std = torch.tensor([self.std], device=comp_device, dtype=comp_type)

        centroid = pos.sum(dim=0, keepdim=True) / n
        if (std == 0).all():
            offset = mean
        else:
            offset = torch.normal(mean=mean, std=std)

        mesh.pos = offset + (pos - centroid)
        return mesh


class ToDevice(object):
    def __init__(self, device: torch.device):
        super().__init__()
        self.argument = device

    def __call__(self, mesh: torch_geometric.data):
        mesh.pos = mesh.pos.to(self.argument)
        mesh.face = mesh.face.to(self.argument)
        mesh.edge_index = mesh.edge_index.to(self.argument)
        mesh.y = mesh.y.to(self.argument)
        return mesh


class Downscaler(object):
    def __init__(self, filename, mesh, factor=4):
        if filename[-4:] != ".npy": filename += ".npy"
        self.downscaled_cache_file = filename
        self._mesh = mesh
        self._E, self._F, self._D = None, None, None
        self.factor = factor

    @property
    def _ds_cached(self):
        return os.path.exists(self.downscaled_cache_file)

    @property
    def _ds_loaded(self):
        return self._D is not None and not self._E is None and not self._F is None

    def _load_transfrom_data(self):
        # if not cached, then compute and store
        if self._ds_cached and self._ds_loaded:
            return
        else:
            if self._ds_cached:  # data is cached, but not loaded (for example after a restart)
                E, F, D = np.load(self.downscaled_cache_file, allow_pickle=True)  # load data
            else:  # data is neither cached nor loaded
                data = self._mesh
                v, f = data.pos.numpy(), data.face.t().numpy()
                _, F, E, D = generate_transform_matrices(v, f, [self.factor] * 3)
                np.save(self.downscaled_cache_file, (E, F, D))

            # assign data to respective fields
            F_t = [torch.tensor(f).t() for f in F]
            D_t = [_scipy_to_torch_sparse(d) for d in D]
            E_t = [_scipy_to_torch_sparse(e) for e in E]
            self._E, self._F, self._D = E_t, F_t, D_t

    @property
    def downscale_matrices(self):
        self._load_transfrom_data()
        return self._D

    @property
    def downscaled_edges(self):
        self._load_transfrom_data()
        return self._E

    @property
    def downscaled_faces(self):
        self._load_transfrom_data()
        return self._F


def _scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape
    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return


class SmalDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self,
                 root: str,
                 device: torch.device = torch.device("cpu"),
                 train: bool = True,
                 test: bool = True,
                 custom: bool = False,
                 custom_list: [int] = [1, 2, 3, 4, 5],
                 transform_data: bool = True):

        self.categories = ["cat", "cow", "dog", "hippo", "horse"]

        # center each mesh into its centroid
        pre_transform = transforms.Center()

        # transform
        if transform_data:
            # rotate and move
            transform = transforms.Compose([
                Move(mean=[0, 0, 0], std=[0.05, 0.05, 0.05]),
                Rotate180(),
                Rotate90(),
                # Rotate(dims=[0, 1, 2]),
                ToDevice(device)])
        else:
            transform = ToDevice(device)

        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.downscaler = Downscaler(
            filename=os.path.join(self.processed_dir, "ds"), mesh=self.get(0), factor=2)

        if train and not test and not custom:
            self.mapping = [i for i in range(len(self)) if self.get(i).pose < 16]
            self.subset = list(range(len(self.mapping)))
            self.data, self.slices = self.collate([self.get(i) for i in range(len(self)) if self.get(i).pose < 16])
        elif not train and test and not custom:
            self.mapping = [i for i in range(len(self)) if self.get(i).pose >= 16]
            self.subset = list(range(len(self.mapping)))
            self.data, self.slices = self.collate([self.get(i) for i in range(len(self)) if self.get(i).pose >= 16])
        elif not train and not test and custom:
            possible_indices = [i for i in range(len(self)) if self.get(i).pose >= 16]  # Whole test set.
            self.mapping = [possible_indices[i] for i in custom_list]
            self.subset = custom_list
            self.data, self.slices = self.collate([self.get(possible_indices[i]) for i in custom_list])

    @property
    def raw_file_names(self):
        files = sorted(os.listdir(self.raw_dir))
        categ_files = [f for f in files if os.path.isfile(os.path.join(self.raw_dir, f)) and f.split(".")[-1] == "ply"]
        return categ_files

    def file_name(self, index):
        r"""Returns file name by mapping its index within the dataset to its original name."""
        assert (index in range(len(self.mapping)))
        map = lambda x: self.raw_file_names[self.mapping[x]]
        return map(index).split(".")[0]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found.')

    def process(self):
        data_list = []
        f2e = transforms.FaceToEdge(remove_faces=False)
        for pindex, path in enumerate(tqdm.tqdm(self.raw_paths)):
            mesh = gio.read_ply(path)
            f2e(mesh)
            tmp = os.path.split(path)[1].split(".")[0].split("_")
            category = tmp[-2]
            model_str, pose_str = tmp[-3], tmp[-1]
            if model_str == 'train':
                mesh.model = 2
            elif model_str == 'val':
                mesh.model = 1
            else:  # model_str == 'test'
                mesh.model = 0
            mesh.pose = int(pose_str)
            mesh.y = self.categories.index(category)
            if self.pre_filter is not None and not self.pre_filter(mesh): continue
            if self.pre_transform is not None: mesh = self.pre_transform(mesh)
            data_list.append(mesh)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def downscale_matrices(self):
        return self.downscaler.downscale_matrices

    @property
    def downscaled_edges(self):
        return self.downscaler.downscaled_edges

    @property
    def downscaled_faces(self):
        return self.downscaler.downscaled_faces

