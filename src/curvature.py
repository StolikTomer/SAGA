import torch
import torch_sparse as tsparse


def tri_areas(vertices, faces):
    v1 = vertices[faces[:, 0], :]
    v2 = vertices[faces[:, 1], :]
    v3 = vertices[faces[:, 2], :]

    v1 = v1 - v3
    v2 = v2 - v3
    return torch.norm(torch.cross(v1, v2, dim=1), dim=1) * .5


def laplacebeltrami_FEM(vertices, faces):
    n = vertices.shape[0]
    m = faces.shape[0]
    device = vertices.device

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

    areas = tri_areas(vertices, faces)
    areas = areas.repeat(6) / 12

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


def laplacebeltrami_FEM_v2(pos, faces):
    if pos.shape[1] != 3: raise ValueError("input position must have shape: [#vertices, 3]")
    if faces.shape[1] != 3: raise ValueError("input faces must have shape [#faces,3]")

    n = pos.shape[0]
    m = faces.shape[0]
    device = pos.device

    angles = {}
    for i in (1.0, 2.0, 3.0):
        a = torch.fmod(torch.as_tensor(i - 1), torch.as_tensor(3.)).long()
        b = torch.fmod(torch.as_tensor(i), torch.as_tensor(3.)).long()
        c = torch.fmod(torch.as_tensor(i + 1), torch.as_tensor(3.)).long()

        ab = pos[faces[:, b], :] - pos[faces[:, a], :];
        ac = pos[faces[:, c], :] - pos[faces[:, a], :];

        ab = torch.nn.functional.normalize(ab, p=2, dim=1)
        ac = torch.nn.functional.normalize(ac, p=2, dim=1)

        # compute the cotangent at the corresponding angle
        o = torch.bmm(ab.view(m, 1, 3), ac.view(m, 3, 1)).squeeze_()  # batched dot product
        o = torch.acos(o)
        o = torch.div(torch.cos(o), torch.sin(o))
        angles[i] = o

    indicesI = torch.cat((faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 1], faces[:, 0]))
    indicesJ = torch.cat((faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 1], faces[:, 0], faces[:, 2]))
    indices = torch.stack((indicesI, indicesJ))

    one_to_n = torch.arange(0, n, dtype=torch.long, device=device)
    eye_indices = torch.stack((one_to_n, one_to_n))

    values = torch.cat((angles[3], angles[1], angles[2], angles[1], angles[3], angles[2])) * 0.5

    areas = tri_areas(pos, faces)
    areas = areas.repeat(6)

    Si, Sv = _stiffness_scatter(indices, values, n)
    ai, av = _lumped_scatter(indices, areas, n)
    return (Si, Sv), (ai, av)


def _stiffness_scatter(indices, cotan, n):
    stiff_i, stiff_v = tsparse.coalesce(indices, -cotan, m=n, n=n, op="add")
    eye_indices = torch.cat((stiff_i[0, :].view(1, -1), stiff_i[0, :].view(1, -1)), dim=0)
    stiff_eye_i, stiff_eye_v = tsparse.coalesce(eye_indices, stiff_v, m=n, n=n, op="add")
    Si = torch.cat((stiff_eye_i, stiff_i), dim=1)
    Sv = torch.cat((-stiff_eye_v, stiff_v))
    return tsparse.coalesce(Si, Sv, n, n)


def _lumped_scatter(indices, areas, n):
    eye_indices = torch.cat((indices[0, :].view(1, -1), indices[0, :].view(1, -1)), dim=0)
    return tsparse.coalesce(eye_indices, areas / 6, m=n, n=n,
                            op="add")  # NOTE I divide by 6 since I count two times (due to how index is created)


def meancurvature(pos, faces):
    if pos.shape[-1] != 3:
        raise ValueError("Vertices positions must have shape [n,3]")

    if faces.shape[-1] != 3:
        raise ValueError("Face indices must have shape [m,3]")

    n = pos.shape[0]
    stiff, mass = laplacebeltrami_FEM_v2(pos, faces)
    ai, av = mass
    mcf = tsparse.spmm(ai, torch.reciprocal(av), n, n, tsparse.spmm(*stiff, n, n, pos))
    return mcf.norm(dim=-1, p=2), stiff, mass


def meancurvature_diff_l2(perturbed_pos, pos, faces):
    ppos = perturbed_pos
    mcp, _, _ = meancurvature(ppos, faces)
    mc, _, (_, a) = meancurvature(pos, faces)
    diff_curvature = mc - mcp
    a = a / a.sum()
    curvature_dist = (a * diff_curvature ** 2).sum().sqrt().item()
    return curvature_dist


def meancurvature_diff_abs(perturbed_pos, pos, faces):
    ppos = perturbed_pos
    mcp, _, _ = meancurvature(ppos, faces)
    mc, _, (_, a) = meancurvature(pos, faces)
    diff_curvature = mc - mcp
    a = a / a.sum()
    curvature_dist = (a * diff_curvature.abs()).sum().item()
    return curvature_dist
