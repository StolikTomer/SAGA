import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
import time

from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import SagaDataset, AttackedDataset
from model import Coma
from transform import Normalize
import pickle


class UnNormalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, vertices):
        assert self.mean is not None and self.std is not None, ('Initialize mean and std to normalize with')
        self.mean = torch.as_tensor(self.mean, dtype=vertices.dtype, device=vertices.device)
        self.std = torch.as_tensor(self.std, dtype=vertices.dtype, device=vertices.device)
        unnormalized_vertices = (vertices - self.mean) / self.std
        return unnormalized_vertices


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay


def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir, opt, lr):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir,
                                        'checkpoint_' + opt + '_' + str(lr) + '_' + str(epoch) + '.pt'))


def main(args):
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)

    print('Initializing parameters')
    template_file_path = os.path.join(os.getcwd(), 'template', 'saga_template.obj')
    assert os.path.exists(template_file_path), 'saga_template.obj file does not exist'

    visuals_train_dir = os.path.join(os.getcwd(), 'visuals_train')
    if not os.path.exists(visuals_train_dir):
        os.mkdir(visuals_train_dir)
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints_saga')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if args.checkpoint_file_name is None:
        checkpoint_file = None
    else:
        checkpoint_file = os.path.join(checkpoint_dir, args.checkpoint_file_name)
        assert os.path.exists(checkpoint_file), 'checkpoint file name does not exist. path: {}'\
                                                .format(checkpoint_file)

    template_mesh = Mesh(filename=template_file_path)
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    batch_size = config['batch_size']
    val_losses = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using device: {}'.format(device))

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Loading Dataset')
    data_dir = os.path.join(os.getcwd(), 'raw_data')
    faces_file = os.path.join(data_dir, 'mesh_faces.npy')
    assert os.path.exists(faces_file), 'mesh faces file does not exist'
    saga_data_file = os.path.join(data_dir, 'coma_FEM.mat')
    assert os.path.exists(saga_data_file), 'saga train data file does not exist'

    faces = np.load(faces_file)
    normalize_transform = Normalize()
    dataset_train = SagaDataset(data_dir, saga_data_file, faces_file, type='train', pre_transform=normalize_transform)
    dataset_test = SagaDataset(data_dir, saga_data_file, faces_file, type='test', pre_transform=normalize_transform)
    unnormalize = UnNormalize((-dataset_train.mean / dataset_train.std).tolist(), (1.0 / dataset_train.std).tolist()) if \
        normalize_transform is not None else None

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    print('Loading model')
    start_epoch = 1
    coma = Coma(dataset_train, config, D_t, U_t, A_t, num_nodes)
    print('optimizer: {}'.format(args.opt))
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(coma.parameters(), lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(coma.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    if checkpoint_file is not None:
        print('checkpoint_file: {}'.format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        coma.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # To find if this is fixed in pytorch
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    coma.to(device)

    if args.eval_attack_flag:
        attack_dir = os.path.join(os.getcwd(), 'attack_data')
        results_file_in = os.path.join(attack_dir, 'attack_data_in.pickle')
        assert os.path.exists(results_file_in), 'attack_data_in.pickle does not exist'
        results_file_out = os.path.join(attack_dir, 'attack_data_out.pickle')

        attack_normalize = Normalize(mean=dataset_train.mean, std=dataset_train.std)
        s_dataset = AttackedDataset(attack_dir, results_file_in, faces_file, type='source', pre_transform=attack_normalize)
        t_dataset = AttackedDataset(attack_dir, results_file_in, faces_file, type='target', pre_transform=attack_normalize)
        adv_dataset = AttackedDataset(attack_dir, results_file_in, faces_file, type='adversary', pre_transform=attack_normalize)

        s_loader = DataLoader(s_dataset, batch_size=1, shuffle=False, num_workers=workers_thread)
        t_loader = DataLoader(t_dataset, batch_size=1, shuffle=False, num_workers=workers_thread)
        adv_loader = DataLoader(adv_dataset, batch_size=1, shuffle=False, num_workers=workers_thread)

        visuals_attack_dir = os.path.join(os.getcwd(), 'visuals_attack')
        if not os.path.exists(visuals_attack_dir):
            os.mkdir(visuals_attack_dir)
        evaluate_attack(coma, visuals_attack_dir, s_loader, t_loader, adv_loader, results_file_in,
                        results_file_out, faces, device, unnormalize=unnormalize, visualize=args.visualize)
        return

    if args.eval_flag:
        print('eval flag is on')
        val_loss = evaluate(coma, visuals_train_dir, test_loader, dataset_test, faces, device, -1, unnormalize=unnormalize,
                            visualize=args.visualize)
        print('val loss', val_loss)
        return

    best_val_loss = float('inf')
    val_loss_history = []

    for epoch in range(start_epoch, total_epochs + 1):
        print("Training for epoch ", epoch)
        start_epoch_time = time.time()
        train_loss = train(coma, train_loader, len(dataset_train), optimizer, device)
        val_loss = evaluate(coma, visuals_train_dir, test_loader, dataset_test, faces, device, epoch, unnormalize=unnormalize,
                            visualize=args.visualize)

        print('epoch ', epoch, ' Train loss ', train_loss, ' Val loss ', val_loss, 'Time(s) ',
              (time.time() - start_epoch_time))
        if val_loss < best_val_loss:
            save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir, args.opt, args.lr)
            best_val_loss = val_loss

        val_loss_history.append(val_loss)
        val_losses.append(best_val_loss)

        if args.opt == 'sgd':
            adjust_learning_rate(optimizer, lr_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train(coma, train_loader, len_dataset, optimizer, device):
    coma.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = coma(data)
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len_dataset


def evaluate(coma, visuals_dir, test_loader, dataset, faces, device, epoch, unnormalize=None, visualize=False):
    coma.eval()
    total_loss = 0
    vis_every_epochs = 100
    vis_every_shapes = 1000 if (epoch >= 0) else 100
    if (epoch > 0) and (epoch % vis_every_epochs != 0):
        visualize = False
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out = coma(data)
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()

        if visualize and i % vis_every_shapes == 0:
            unnorm_out = unnormalize(out)
            unnorm_y = unnormalize(data.y)
            save_out = unnorm_out.detach().cpu().numpy()
            expected_out = (unnorm_y.detach().cpu().numpy())
            result_mesh = Mesh(v=save_out, f=faces)
            expected_mesh = Mesh(v=expected_out, f=faces)
            meshviewer = MeshViewers(shape=(1, 2))
            meshviewer[0][0].set_dynamic_meshes([result_mesh])
            meshviewer[0][1].set_dynamic_meshes([expected_mesh])
            meshviewer[0][0].save_snapshot(os.path.join(visuals_dir, 'file' + str(i) + '.png'), blocking=False)

    return total_loss / len(dataset)


def evaluate_attack(model, visuals_dir, s_loader, t_loader, adv_loader, results_file_in, results_file_out,
                    faces, device, unnormalize=None, visualize=False):
    model.eval()

    dict_out_list = []
    with open(results_file_in, 'rb') as handle:
        pair_dicts_in = pickle.load(handle)
    pair_dicts_in = [item for item in pair_dicts_in if (item["step"] == 499)]
    for i, (s_data, t_data, adv_data) in enumerate(zip(s_loader, t_loader, adv_loader)):
        idx_start_time = time.time()
        s_data = s_data.to(device)
        t_data = t_data.to(device)
        adv_data = adv_data.to(device)

        with torch.no_grad():
            s_recon = model(s_data)
            t_recon = model(t_data)
            adv_recon = model(adv_data)

        s_verts = unnormalize(s_data.y)
        t_verts = unnormalize(t_data.y)
        adv_verts = unnormalize(adv_data.y)
        s_recon = unnormalize(s_recon)
        t_recon = unnormalize(t_recon)
        adv_recon = unnormalize(adv_recon)

        s_verts_np = s_verts.detach().cpu().numpy()
        t_verts_np = t_verts.detach().cpu().numpy()
        adv_verts_np = adv_verts.detach().cpu().numpy()
        s_recon_np = s_recon.detach().cpu().numpy()
        t_recon_np = t_recon.detach().cpu().numpy()
        adv_recon_np = adv_recon.detach().cpu().numpy()

        vis_indices = [4220, 4784, 2268, 1609, 1651, 443, 4034]
        if visualize and i in vis_indices:
            print('visualizing source index:{}...'.format(i))
            s_mesh = Mesh(v=s_verts_np, f=faces)
            s_recon_mesh = Mesh(v=s_recon_np, f=faces)
            meshviewer = MeshViewers(shape=(1, 2))
            meshviewer[0][0].set_dynamic_meshes([s_mesh])
            meshviewer[0][1].set_dynamic_meshes([s_recon_mesh])
            meshviewer[0][0].save_snapshot(os.path.join(visuals_dir, 'attack_s_' + str(i) + '.png'), blocking=False)

            print('visualizing target index:{}...'.format(i))
            t_mesh = Mesh(v=t_verts_np, f=faces)
            t_recon_mesh = Mesh(v=t_recon_np, f=faces)
            meshviewer = MeshViewers(shape=(1, 2))
            meshviewer[0][0].set_dynamic_meshes([t_mesh])
            meshviewer[0][1].set_dynamic_meshes([t_recon_mesh])
            meshviewer[0][0].save_snapshot(os.path.join(visuals_dir, 'attack_t_' + str(i) + '.png'), blocking=False)

            print('visualizing adversary index:{}...'.format(i))
            adv_mesh = Mesh(v=adv_verts_np, f=faces)
            adv_recon_mesh = Mesh(v=adv_recon_np, f=faces)
            meshviewer = MeshViewers(shape=(1, 2))
            meshviewer[0][0].set_dynamic_meshes([adv_mesh])
            meshviewer[0][1].set_dynamic_meshes([adv_recon_mesh])
            meshviewer[0][0].save_snapshot(os.path.join(visuals_dir, 'attack_adv_' + str(i) + '.png'), blocking=False)

        step = pair_dicts_in[i]["step"]
        s_label = pair_dicts_in[i]["s_label"]
        t_label = pair_dicts_in[i]["t_label"]

        dict_out = {"step": step, "pair_number": i, "s_label": s_label, "t_label": t_label,
                    "s_mesh": s_verts_np, "t_mesh": t_verts_np,
                    "adv_mesh": adv_verts_np, "adv_recon_mesh": adv_recon_np,
                    "s_recon_mesh": s_recon_np, "t_recon_mesh": t_recon_np}

        dict_out_list.append(dict_out)
        print('Finished processing index: {}, time(s): {}...'.format(i, (time.time() - idx_start_time)))
    print('saving out dictionary...')
    with open(results_file_out, 'wb') as handle:
        pickle.dump(dict_out_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def type2bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, int) or isinstance(value, float):
        return bool(value)
    elif isinstance(value, str):
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    else:
        raise argparse.ArgumentTypeError('possible boolean indications should be of type int, float or str')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', default=None, help='path of config file')
    parser.add_argument('--lr', default='0.008', type=float, help='learning rate')
    parser.add_argument('--opt', default='sgd', help='sgd, adam')
    parser.add_argument('--eval_flag', default=0, help='evaluate the model')
    parser.add_argument('--eval_attack_flag', default=0, help='evaluation of attacked shapes')
    parser.add_argument('--visualize', default=1, help='including visualizations in the run')
    parser.add_argument('--checkpoint_file_name', default=None, help='path to the learned weights')

    args = parser.parse_args()
    args.visualize = type2bool(args.visualize)
    args.eval_attack_flag = type2bool(args.eval_attack_flag)
    args.eval_flag = type2bool(args.eval_flag)

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
