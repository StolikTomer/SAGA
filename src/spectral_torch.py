import torch
import scipy.sparse as scisparse
from scipy.sparse import linalg as sla
import time
from utils import log_string


def calc_tri_areas(vertices, faces):
    v1 = vertices[faces[:, 0], :]
    v2 = vertices[faces[:, 1], :]
    v3 = vertices[faces[:, 2], :]

    v1 = v1 - v3
    v2 = v2 - v3
    return torch.norm(torch.cross(v1, v2, dim=1), dim=1) * .5


def calc_LB_FEM(vertices, faces, device):
    n = vertices.shape[0]

    angles = {}
    for i in (1.0, 2.0, 3.0):
        a = torch.fmod(torch.as_tensor(i - 1), torch.as_tensor(3.)).long()
        b = torch.fmod(torch.as_tensor(i), torch.as_tensor(3.)).long()
        c = torch.fmod(torch.as_tensor(i + 1), torch.as_tensor(3.)).long()

        ab = vertices[faces[:, b], :] - vertices[faces[:, a], :];
        ac = vertices[faces[:, c], :] - vertices[faces[:, a], :];

        ab = torch.nn.functional.normalize(ab, p=2, dim=1)
        ac = torch.nn.functional.normalize(ac, p=2, dim=1)

        o = torch.mul(ab, ac)
        o = torch.sum(o, dim=1)
        o = torch.acos(o)
        o = torch.div(torch.cos(o), torch.sin(o))

        angles[i] = o

    indicesI = torch.cat((faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 1], faces[:, 0]))
    indicesJ = torch.cat((faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 1], faces[:, 0], faces[:, 2]))
    indices = torch.stack((indicesI, indicesJ))

    one_to_n = torch.arange(0, n, dtype=torch.long, device=device)
    eye_indices = torch.stack((one_to_n, one_to_n))

    values = torch.cat((angles[3], angles[1], angles[2], angles[1], angles[3], angles[2])) * 0.5

    stiff = torch.sparse_coo_tensor(indices=indices, dtype=values.dtype,
                                    values=-values,
                                    device=device,
                                    size=(n, n)).coalesce()
    stiff = stiff + torch.sparse_coo_tensor(indices=eye_indices, dtype=values.dtype,
                                            values=-torch.sparse.sum(stiff, dim=0).to_dense(),
                                            device=device,
                                            size=(n, n)).coalesce()

    areas = calc_tri_areas(vertices, faces)
    areas = areas.repeat(6) / 12.

    mass = torch.sparse_coo_tensor(indices=indices, dtype=values.dtype,
                                   values=areas,
                                   device=device,
                                   size=(n, n)).coalesce()
    mass = mass + torch.sparse_coo_tensor(indices=eye_indices, dtype=values.dtype,
                                          values=torch.sparse.sum(mass, dim=0).to_dense(),
                                          device=device,
                                          size=(n, n)).coalesce()

    lumped_mass = torch.sparse.sum(mass, dim=1).to_dense()

    return stiff, mass, lumped_mass


def sparse_dense_mul(s, d):
    s = s.coalesce()
    i = s.indices()
    v = s.values()
    dv = d[i[0, :], i[1, :]]
    return torch.sparse.FloatTensor(i, v * dv, s.size()).coalesce()


def decomposition_torch(stiff, lumped_mass):
    # Cholesky decomposition for diagonal matrices
    lower = torch.sqrt(lumped_mass)
    # Compute inverse
    lower_inv = 1 / lower
    C = sparse_dense_mul(stiff, lower_inv[None, :] * lower_inv[:, None])
    return C


def eigsh(values, indices, k, sigma=-1e-5):
    values = values.detach().cpu().numpy()
    indices = indices.detach().cpu().numpy()

    Ascipy = scisparse.coo_matrix((values, indices)).tocsc()
    e, phi = sla.eigsh(Ascipy, k, sigma=sigma)

    return e, phi


def calculate_eigenvalues_batch(vertices, faces, evals_num, device, evects_num,
                                log_file=None, return_vectors=False):
    max_eigen_num = max(evals_num, evects_num)
    evalues_batch = []
    evectors_batch = []
    for shape_idx in range(vertices.shape[0]):
        start_idx_time = time.time()
        W, _, A = calc_LB_FEM(vertices[shape_idx], faces, device=device)
        C = decomposition_torch(W, A)

        try:
            numpy_eigvals, numpy_eigvectors = eigsh(C.values(), C.indices(), max_eigen_num + 1)
        except:
            log_string(log_file, 'eigsh returned None value (failed), skipping shape_idx={}'.format(shape_idx))
            continue

        A_inverse = A.rsqrt().detach().cpu().numpy()

        numpy_eigvectors_normalized = A_inverse[:, None] * numpy_eigvectors

        if abs(numpy_eigvals[0]) < 1:  # the first eigenvalue should be ~0
            evalues = torch.tensor(numpy_eigvals[0:evals_num]).to(device)
            evectors = torch.tensor(numpy_eigvectors_normalized[:, 0:evects_num]).to(device)
        else:
            evalues = torch.tensor(numpy_eigvals[1:(evals_num + 1)]).to(device)
            evectors = torch.tensor(numpy_eigvectors_normalized[:, 1:(evects_num + 1)]).to(device)
        evectors_batch.append(evectors)

        evalues_batch.append(evalues)
        end_idx_time = time.time()
        idx_calculations_duration = (end_idx_time - start_idx_time)
        log_string(log_file, 'Calculating eigenvalues for index %04d out of %d, duration: %.2f seconds' %
                   (shape_idx, len(vertices), idx_calculations_duration))

    if not evalues_batch:
        log_string(log_file, 'failed to find eigenvalues for all the shapes in the batch, returning None')
        if return_vectors:
            return None, None
        else:
            return None
    evals_batch_torch = torch.stack(evalues_batch)
    if return_vectors:
        evectors_batch_torch = torch.stack(evectors_batch)
        return evals_batch_torch.to(device), evectors_batch_torch.to(device)
    return evals_batch_torch.to(device)
