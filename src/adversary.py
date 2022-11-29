import torch
from utils_attack import get_barycenter_matrix
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D


class Adversary:
    def __init__(self, params, num_vertices, device, dtype=torch.float, log_file=None):
        self.device = device
        self.dtype = dtype
        self._num_vertices = num_vertices
        self._num_evals = params['N_evals']
        self._num_evects = params['N_evects']
        self.log_file = log_file
        self._shape_delta = None
        self._shape_beta = None
        self.pert_delta = None  # delta = Euclidean perturbation
        self.pert_beta = None  # beta = spectral perturbation
        self.barycenter_mat = None

    def init_pert(self, batch_size, faces):
        self._shape_delta = [batch_size, self._num_vertices, 3]
        self._shape_beta = [batch_size, self._num_evects, 3]
        self.pert_delta = torch.zeros([batch_size, self._num_vertices, 3],
                                      device=self.device,
                                      dtype=self.dtype,
                                      requires_grad=True)
        self.pert_beta = torch.zeros([batch_size, self._num_evects, 3],
                                     device=self.device,
                                     dtype=self.dtype,
                                     requires_grad=True)
        self.barycenter_mat = get_barycenter_matrix(self._num_vertices, faces).to(self.device)

    def get_pert_delta(self):
        return self.pert_delta

    def get_pert_beta(self):
        return self.pert_beta

    def get_pert_delta_np(self):
        return self.pert_delta.detach().cpu().numpy()

    def get_pert_beta_np(self):
        return self.pert_beta.detach().cpu().numpy()

    def spatial_attack(self, params, input_mesh, evectors=None, alphas=None):
        if params['Adversary_type'] == 'beta':  # each shape is defined by its spectral-coefficient (alphas)
            num_evects = params['N_evects']
            tot_evects = alphas.shape[1]
            if params['weights_on_evects'] == 'low':  # low frequencies
                if params['pert_type'] == 'mul':
                    pert_alphas = alphas[:, :num_evects, :] * (1 + self.pert_beta)
                else:  # params['pert_type'] == 'add'
                    pert_alphas = alphas[:, :num_evects, :] + self.pert_beta
                pert_evects = evectors[:, :, :num_evects]

                unpert_alphas = alphas[:, num_evects:, :]
                unpert_evects = evectors[:, :, num_evects:]
            else:  # params['weights_on_evects'] == 'high':
                if params['pert_type'] == 'mul':
                    pert_alphas = alphas[:, (tot_evects - num_evects):, :] * (1 + self.pert_beta)
                else:  # params['pert_type'] == 'add'
                    pert_alphas = alphas[:, (tot_evects - num_evects):, :] + self.pert_beta
                pert_evects = evectors[:, :, (tot_evects - num_evects):]

                unpert_alphas = alphas[:, :(tot_evects - num_evects), :]
                unpert_evects = evectors[:, :, :(tot_evects - num_evects)]

            adv_mesh = torch.matmul(pert_evects, pert_alphas) + torch.matmul(unpert_evects, unpert_alphas)
        else:  # params['Adversary_type'] == 'delta' - Euclidean attack
            adv_mesh = input_mesh + self.pert_delta
        return adv_mesh

    def get_delta_loss(self, weights, sqrt=False, return_max=False):
        w = weights['W_reg_spat']
        if w == 0:
            return torch.tensor(0.)

        delta_square_per_vertex = torch.mean(torch.square(self.pert_delta), dim=2)
        delta_square_max_batch, _ = torch.max(delta_square_per_vertex, dim=1)
        delta_square_batch = torch.mean(delta_square_per_vertex, dim=1)
        if sqrt:
            delta_square_root_batch = torch.sqrt(delta_square_batch)
            delta_reg_loss = torch.mean(delta_square_root_batch)
            delta_square_root_max_batch = torch.sqrt(delta_square_max_batch)
            delta_reg_max_loss = torch.mean(delta_square_root_max_batch)
        else:
            delta_reg_loss = torch.mean(delta_square_batch)
            delta_reg_max_loss = torch.mean(delta_square_max_batch)

        if return_max:
            return delta_reg_loss * w, delta_reg_max_loss
        else:
            return delta_reg_loss * w

    def get_beta_loss(self, weights, sqrt=False, return_max=False):
        w = weights['W_reg_spat']
        if w == 0:
            return torch.tensor(0.)

        beta_square_per_evect = torch.mean(torch.square(self.pert_beta), dim=2)
        beta_square_max_batch, _ = torch.max(beta_square_per_evect, dim=1)
        beta_square_batch = torch.mean(beta_square_per_evect, dim=1)
        if sqrt:
            beta_square_root_batch = torch.sqrt(beta_square_batch)
            beta_reg_loss = torch.mean(beta_square_root_batch)
            beta_square_root_max_batch = torch.sqrt(beta_square_max_batch)
            beta_reg_max_loss = torch.mean(beta_square_root_max_batch)
        else:
            beta_reg_loss = torch.mean(beta_square_batch)
            beta_reg_max_loss = torch.mean(beta_square_max_batch)

        if return_max:
            return beta_reg_loss * w, beta_reg_max_loss
        else:
            return beta_reg_loss * w

    @staticmethod
    def get_mse_loss(weights, mesh_1, mesh_2):
        w = weights['W_recon_mse']
        if w == 0:
            return torch.tensor(0.)

        mse = torch.nn.MSELoss()
        loss_recon_mse = mse(mesh_1, mesh_2)
        return loss_recon_mse * w

    def get_barycenter_loss(self, weights, mesh_1, mesh_2):
        w = weights['W_reg_bary']
        if w == 0:
            return torch.tensor(0.)

        loss_bary = 0
        for shape_idx in range(mesh_1.shape[0]):
            mesh_diff = mesh_2 - mesh_1
            loss_bary += torch.mean((torch.matmul(self.barycenter_mat, mesh_diff[shape_idx])).pow(2))
        return loss_bary * w

    @staticmethod
    def get_edge_loss(weights, mesh, adv_mesh, faces):
        w = weights['W_reg_edge']
        if w == 0:
            return torch.tensor(0.)

        loss_edges = 0
        for shape_idx in range(mesh.shape[0]):
            vertices = mesh[shape_idx]
            pert_vertices = adv_mesh[shape_idx]

            edge_0_clean = torch.norm((vertices[faces[:, 0], :] - vertices[faces[:, 1], :]), dim=1)
            edge_1_clean = torch.norm((vertices[faces[:, 1], :] - vertices[faces[:, 2], :]), dim=1)
            edge_2_clean = torch.norm((vertices[faces[:, 2], :] - vertices[faces[:, 0], :]), dim=1)

            edge_0_pert = torch.norm((pert_vertices[faces[:, 0], :] - pert_vertices[faces[:, 1], :]), dim=1)
            edge_1_pert = torch.norm((pert_vertices[faces[:, 1], :] - pert_vertices[faces[:, 2], :]), dim=1)
            edge_2_pert = torch.norm((pert_vertices[faces[:, 2], :] - pert_vertices[faces[:, 0], :]), dim=1)

            face_edge_norm = (1 / 6) * (torch.mean((edge_0_clean - edge_0_pert).pow(2)) +
                                        torch.mean((edge_1_clean - edge_1_pert).pow(2)) +
                                        torch.mean((edge_2_clean - edge_2_pert).pow(2)))

            loss_edges += face_edge_norm

        return loss_edges * w

    def get_area_loss(self, weights, mesh_1, mesh_2, faces, focus_on_details=True):
        w = weights['W_reg_area']
        if w == 0:
            return torch.tensor(0.)

        mesh_diff = mesh_2 - mesh_1
        area_weights = torch.zeros(size=mesh_1.shape).to(self.device)
        for shape_idx in range(mesh_1.shape[0]):
            vertices = mesh_1[shape_idx]
            edge_0_clean = vertices[faces[:, 0], :] - vertices[faces[:, 1], :]
            edge_1_clean = vertices[faces[:, 1], :] - vertices[faces[:, 2], :]
            areas_clean = torch.norm(torch.cross(edge_0_clean, edge_1_clean, dim=1), dim=1) * .5

            area_weights[shape_idx, faces[:, 0], :] += areas_clean.repeat(3, 1).t()
            area_weights[shape_idx, faces[:, 1], :] += areas_clean.repeat(3, 1).t()
            area_weights[shape_idx, faces[:, 2], :] += areas_clean.repeat(3, 1).t()

            if focus_on_details:  # big weights for vertices that affect small areas
                area_weights[shape_idx] = 1 / area_weights[shape_idx]
            min_val = torch.min(area_weights[shape_idx, :, 0])
            max_val = torch.max(area_weights[shape_idx, :, 0])
            area_weights[shape_idx] = (area_weights[shape_idx] - min_val) / (max_val - min_val)

        return torch.mean((area_weights * mesh_diff).pow(2)) * w

    @staticmethod
    def get_normals_loss(weights, mesh_1, mesh_2, faces):
        w = weights['W_reg_normals']
        if w == 0:
            return torch.tensor(0.)

        loss_normals = 0
        for shape_idx in range(mesh_1.shape[0]):
            vertices = mesh_1[shape_idx]
            pert_vertices = mesh_2[shape_idx]

            edge_0_clean = vertices[faces[:, 0], :] - vertices[faces[:, 1], :]
            edge_1_clean = vertices[faces[:, 1], :] - vertices[faces[:, 2], :]
            normals_clean = torch.cross(edge_0_clean, edge_1_clean, dim=1)
            normal_norms_clean = torch.norm(normals_clean, dim=1)
            normals_clean = torch.div(normals_clean, normal_norms_clean.repeat(3, 1).t())

            edge_0_pert = pert_vertices[faces[:, 0], :] - pert_vertices[faces[:, 1], :]
            edge_1_pert = pert_vertices[faces[:, 1], :] - pert_vertices[faces[:, 2], :]
            normals_pert = torch.cross(edge_0_pert, edge_1_pert, dim=1)
            normal_norms_pert = torch.norm(normals_pert, dim=1)
            normals_pert = torch.div(normals_pert, normal_norms_pert.repeat(3, 1).t())

            loss_normals += torch.mean((normals_clean - normals_pert).pow(2))

        return loss_normals * w

    @staticmethod
    def get_chamfer_loss(weights, mesh_1, mesh_2):
        w = weights['W_reg_chamfer']
        if w == 0:
            return torch.tensor(0.)

        dist_chamfer = dist_chamfer_3D.chamfer_3DDist()
        dist1, dist2, idx1, idx2 = dist_chamfer(mesh_1, mesh_2)
        chamfer_loss = torch.mean(dist1) + torch.mean(dist2)
        return chamfer_loss * w
