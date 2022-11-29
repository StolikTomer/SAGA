import os
import torch
import numpy as np
import pickle
import datetime
from utils import get_directories, get_device, log_string, get_file, visualize_mesh_subplots, get_argument_parser
from utils_models import load_trained_classifier, get_classifier_params
from utils_attack import get_results_path
from plotly_visualization import visualize_and_compare
from train_classifier import evaluate

flags = get_argument_parser()

# ---------- Configurations ----------
dataset = flags.dataset
result_type = flags.result_type
visualize = flags.visualize
visualize_mistakes = flags.visualize_mistakes
print_confusion_matrix = flags.print_confusion_matrix
show_src_heatmap = flags.show_src_heatmap

data_dir, models_dir, logs_dir, results_dir, images_dir = get_directories(dataset=dataset)
classifier_dir = os.path.join(models_dir, 'classifiers')
vis_list = []

params = get_classifier_params()
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = get_file(logs_dir, 'evaluate_classifier_' + dataset + '_' + current_time + '.txt')
device = get_device(log_file=log_file)

results_file = get_results_path(dataset, results_dir, result_type)
log_string(log_file, 'result_type: {}\nresults file: {}'.format(result_type, results_file))
with open(results_file, 'rb') as handle:
    all_pair_dict_list = pickle.load(handle)

save_faces_file = os.path.join(data_dir, 'raw', 'mesh_faces.npy')
assert os.path.exists(save_faces_file), 'saved faces file was not found'
mesh_faces = np.load(save_faces_file)

# ---------- Considering only the end-of-attack savings ----------
attack_steps = 500 if dataset == 'coma' else 3000
pair_dict_list = [item for item in all_pair_dict_list if
                  (item["step"] == attack_steps - 1)]

log_string(log_file, 'dataset: {}, result_type: {}, number of pairs: {}, results_file:\n{}'
           .format(dataset, result_type, len(pair_dict_list), results_file))

s_accuracy_list = []
adv_accuracy_list = []
t_accuracy_list = []
s_recon_accuracy_list = []
adv_recon_accuracy_list = []
adv_recon_untargeted_accuracy_list = []
t_recon_accuracy_list = []

# ---------- Loading saved values ----------
mesh_faces_torch = torch.from_numpy(mesh_faces).to(device)
s_mesh_list = [torch.from_numpy(item["s_mesh"]).to(device) for item in pair_dict_list]
t_mesh_list = [torch.from_numpy(item["t_mesh"]).to(device) for item in pair_dict_list]
adv_mesh_list = [torch.from_numpy(item["adv_mesh"]).to(device) for item in pair_dict_list]
s_recon_mesh_list = [torch.from_numpy(item["s_recon_mesh"]).to(device) for item in pair_dict_list]
t_recon_mesh_list = [torch.from_numpy(item["t_recon_mesh"]).to(device) for item in pair_dict_list]
adv_recon_mesh_list = [torch.from_numpy(item["adv_recon_mesh"]).to(device) for item in pair_dict_list]
s_label_list = [item["s_label"] for item in pair_dict_list]
t_label_list = [item["t_label"] for item in pair_dict_list]

# ---------- Loading the pre-trained classifier ----------
classifier = load_trained_classifier(params, classifier_dir, device, dataset)

t_accuracy, t_confusion = evaluate(t_mesh_list, t_label_list, classifier, faces=mesh_faces, visualize=visualize,
                                   vis_list=vis_list, visualize_mistakes=visualize_mistakes, log_file=log_file)
log_string(log_file, "\nClean target classification accuracy: {}\n".format(t_accuracy))
if print_confusion_matrix:
    log_string(log_file, "\nClean target classification confusion:\n{}\n".format(t_confusion))

t_recon_accuracy, t_recon_confusion = evaluate(t_recon_mesh_list, t_label_list, classifier,
                                               faces=mesh_faces, visualize=visualize, vis_list=vis_list,
                                               visualize_mistakes=visualize_mistakes, log_file=log_file)
log_string(log_file, "\nClean target-recon classification accuracy: {}\n".format(t_recon_accuracy))
if print_confusion_matrix:
    log_string(log_file, "\nClean target-recon classification confusion:\n{}\n".format(t_recon_confusion))

adv_recon_accuracy, adv_recon_untargeted_accuracy, adv_recon_confusion = \
    evaluate(adv_recon_mesh_list, t_label_list, classifier, src_labels=s_label_list, faces=mesh_faces,
             visualize=visualize, vis_list=vis_list, visualize_mistakes=visualize_mistakes, log_file=log_file)
log_string(log_file, "\nAdv-recon classification accuracy: {}, untargeted accuracy: {}"
           .format(adv_recon_accuracy, adv_recon_untargeted_accuracy))
if print_confusion_matrix:
    log_string(log_file, "\nAdv-recon classification confusion:\n{}\n".format(adv_recon_confusion))

if visualize:
    for pair_number in vis_list:
        pair_dict = [item for item in pair_dict_list if (item["pair_number"] == pair_number)][0]
        s_mesh = pair_dict["s_mesh"]
        t_mesh = pair_dict["t_mesh"]
        adv_mesh = pair_dict["adv_mesh"]
        s_recon_mesh = pair_dict["s_recon_mesh"]
        t_recon_mesh = pair_dict["t_recon_mesh"]
        adv_recon_mesh = pair_dict["adv_recon_mesh"]

        visualize_mesh_subplots(s_mesh, adv_mesh, mesh_faces, title_1='src_' + str(pair_number),
                                title_2='adv_' + str(pair_number))
        visualize_mesh_subplots(t_mesh, adv_recon_mesh, mesh_faces, title_1='target_' + str(pair_number),
                                title_2='adv_recon_' + str(pair_number))

        if show_src_heatmap:
            src_v = torch.from_numpy(s_mesh)
            adv_v = torch.from_numpy(adv_mesh)
            faces_v = torch.from_numpy(mesh_faces)
            visualize_and_compare(adv_v, faces_v, src_v, faces_v,
                                  (src_v - adv_v).norm(p=2, dim=-1))


