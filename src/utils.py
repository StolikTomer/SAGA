import open3d as o3d
import torch
import os
import argparse


def visualize_mesh_and_pc(mesh_vertices, mesh_triangles, pc_points=None, colors=None, window_name='Open3D'):
    T = o3d.geometry.TriangleMesh()
    T.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    T.triangles = o3d.utility.Vector3iVector(mesh_triangles)
    T.compute_vertex_normals()
    if colors is not None:
        T.vertex_colors = o3d.utility.Vector3dVector(colors)
    vis_list = [T]

    if pc_points is not None:
        PC = o3d.geometry.PointCloud()
        PC.points = o3d.utility.Vector3dVector(pc_points)
        vis_list.append(PC)

    o3d.visualization.draw_geometries(vis_list, window_name=window_name)


def visualize_mesh_subplots(vertices_1, vertices_2, mesh_triangles, title_1='First Mesh', title_2='Second Mesh',
                            save_file_1=None, save_file_2=None):
    T1 = o3d.geometry.TriangleMesh()
    T1.vertices = o3d.utility.Vector3dVector(vertices_1)
    T1.triangles = o3d.utility.Vector3iVector(mesh_triangles)
    T1.compute_vertex_normals()

    T2 = o3d.geometry.TriangleMesh()
    T2.vertices = o3d.utility.Vector3dVector(vertices_2)
    T2.triangles = o3d.utility.Vector3iVector(mesh_triangles)
    T2.compute_vertex_normals()

    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window(window_name=title_1, width=960, height=540, left=0, top=0)
    vis1.add_geometry(T1)

    vis2 = o3d.visualization.VisualizerWithEditing()
    vis2.create_window(window_name=title_2, width=960, height=540, left=960, top=0)
    vis2.add_geometry(T2)

    while True:
        vis1.update_geometry(T1)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
        if save_file_1 is not None:
            vis1.capture_screen_image(save_file_1)

        vis2.update_geometry(T2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
        if save_file_2 is not None:
            vis2.capture_screen_image(save_file_2)

    vis1.destroy_window()
    vis2.destroy_window()


def log_string(log_file, log_str):
    if not log_file:
        return
    log_file.write(log_str + '\n')
    log_file.flush()
    print(log_str)


def get_file(file_dir, file_name, wr='w'):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    return open(os.path.join(file_dir, file_name), wr)


def get_device(log_file=None, use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        gpu_num = 0
        log_string(log_file, 'cuda gpu is available. torch.cuda.current_device(): {}'
                   .format(torch.cuda.current_device()))
        torch.cuda.set_device(gpu_num)
        device = 'cuda:' + str(gpu_num)
    else:
        log_string(log_file, 'avoiding cuda gpu, choosing cpu device')
        device = 'cpu'
    return device


def get_directory(parent_dir, dir_name, dataset):
    directory = os.path.join(parent_dir, dir_name)
    if not os.path.exists(directory):
        os.mkdir(directory)
    sub_directory = os.path.join(directory, dataset)
    if not os.path.exists(sub_directory):
        os.mkdir(sub_directory)
    return sub_directory


def get_directories(dataset):
    project_dir = os.path.join(os.getcwd(), os.pardir)

    data_dir = get_directory(project_dir, dataset, 'data')
    models_dir = get_directory(project_dir, dataset, 'models')
    logs_dir = get_directory(project_dir, dataset, 'logs')
    results_dir = get_directory(project_dir, dataset, 'results')
    images_dir = get_directory(project_dir, dataset, 'images')

    return data_dir, models_dir, logs_dir, results_dir, images_dir


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


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coma',
                        help='coma, smal')
    parser.add_argument('--learning_rate', default=None,
                        help='learning rate for the attack perturbation')
    parser.add_argument('--attack_steps', type=int, default=None,
                        help='number of attack steps')
    parser.add_argument('--num_evals', type=int, default=30,
                        help='number of eigenvalues')
    parser.add_argument('--num_evects', type=int, default=None,
                        help='number of eigenvectors')
    parser.add_argument('--pert_type', default=None,
                        help='additive (add) or multiplicative (mul) perturbation')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed')
    parser.add_argument('--save_results', default=1,
                        help='save results')
    parser.add_argument('--visualize', default=0,
                        help='show visualizations and plots')
    parser.add_argument('--show_src_heatmap', default=0,
                        help='show heatmaps on the adversarial shape')
    parser.add_argument('--reduced_memory_mode', type=str, default='weak',
                        help='reduced number of shapes (strong, weak, none) in coma dataset')
    parser.add_argument('--purpose', type=str, default='test',
                        help='train, val, test')
    parser.add_argument('--attack_batch_size', type=int, default=1,
                        help='batch size for the attack')
    parser.add_argument('--adversary_type', type=str, default='beta',
                        help='spectral (beta) or Euclidean (delta) perturbation type')
    parser.add_argument('--weights_on_evects', type=str, default='low',
                        help='perturbing the low/high range of frequencies')
    parser.add_argument('--num_src_per_class', type=int, default=50,
                        help='number of source shapes from each class')
    parser.add_argument('--num_targets_per_src', type=int, default=1,
                        help='number of targets for each source')
    parser.add_argument('--random_targets_mode', default=0,
                        help='choose targets randomly and not by closest shape in the target class')
    parser.add_argument('--inter_class_pairs', default=1,
                        help='source-target pairs are picked from different classes (inter class -True) or'
                             'from the same class (intra class -False)')
    parser.add_argument('--use_self_evects', default=0,
                        help='perturb the self-eigenvectors of the source shape (avoid using the shared basis)')
    parser.add_argument('--num_shared_evects', type=int, default=3000,
                        help='number of eigenvectors in the shared basis')
    parser.add_argument('--check_saving', default=0,
                        help='load the saved data and print it')
    parser.add_argument('--result_type', type=str, default='saga',
                        help='saga, pc, delta, oods, oodt, coma_transfer,'
                             'mlp_transfer, self_evects, random_targets')
    parser.add_argument('--evaluation_type', type=str, default='visual',
                        help='visual, beta, tsne, freq_ablation, curv_dist')
    parser.add_argument('--save_images', default=0,
                        help='save every plot instead of showing it')
    parser.add_argument('--image_suffix', type=str, default='.png',
                        help='.png or .pdf')
    parser.add_argument('--visualize_mistakes', default=0,
                        help='show visualizations of mistakes')
    parser.add_argument('--print_confusion_matrix', default=0,
                        help='print confusion matrix')
    parser.add_argument('--save_ae_every', type=int, default=100,
                        help='save model after every x epochs')
    parser.add_argument('--classifier', type=str, default='PointNet',
                        help='FC, PointNet')
    parser.add_argument('--detector_test_class', type=int, default=0,
                        help='the index of the class used for val and test')
    parser.add_argument('--detector', type=str, default='FC',
                        help='FC, PointNet')
    parser.add_argument('--stability_step', type=int, default=0,
                        help='supported values are 0, 1, 2, 3')

    parser.add_argument('--w_recon_mse', type=float, default=0,
                        help='weight for mse reconstruction loss')
    parser.add_argument('--w_reg_spat', type=float, default=0,
                        help='Weight for spatial l2 regularization loss')
    parser.add_argument('--w_reg_bary', type=float, default=0,
                        help='weight for bary regularization loss')
    parser.add_argument('--w_reg_edge', type=float, default=0,
                        help='weight for edge regularization loss')
    parser.add_argument('--w_reg_area', type=float, default=0,
                        help='weight for area regularization loss')
    parser.add_argument('--w_reg_normals', type=float, default=0,
                        help='weight for normals regularization loss')
    parser.add_argument('--w_reg_chamfer', type=float, default=0,
                        help='weight for chamfer regularization loss')

    flags = parser.parse_args()

    flags.visualize = type2bool(flags.visualize)
    flags.show_src_heatmap = type2bool(flags.show_src_heatmap)
    flags.save_results = type2bool(flags.save_results)
    flags.random_targets_mode = type2bool(flags.random_targets_mode)
    flags.inter_class_pairs = type2bool(flags.inter_class_pairs)
    flags.use_self_evects = type2bool(flags.use_self_evects)
    flags.check_saving = type2bool(flags.check_saving)
    flags.save_images = type2bool(flags.save_images)
    flags.visualize_mistakes = type2bool(flags.visualize_mistakes)
    flags.print_confusion_matrix = type2bool(flags.print_confusion_matrix)

    if flags.learning_rate is None:
        flags.learning_rate = 0.0001 if flags.dataset == 'coma' else 0.01
    if flags.attack_steps is None:
        flags.attack_steps = 500 if flags.dataset == 'coma' else 3000
    if flags.num_evects is None:
        flags.num_evects = 500 if flags.dataset == 'coma' else 2000
    if flags.pert_type is None:
        flags.pert_type = 'add' if flags.dataset == 'coma' else 'mul'

    return flags
