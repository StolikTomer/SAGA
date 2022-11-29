import os
import torch
import numpy as np
import pickle
import datetime
from matplotlib import pyplot as plt
from utils import get_directories, get_device, log_string, get_file, get_argument_parser
from utils_data import Data
from utils_models import get_detector, get_detector_params, get_detector_optimizer, load_trained_detector
from utils_attack import get_attack_params, get_results_path
from train_classifier import train, evaluate

# ---------- Configurations ----------
debug_mode = False
flags = get_argument_parser()

params = get_detector_params()
if debug_mode:
    rand_seed = 1
    dataset = 'coma'
    purpose = 'test'
    result_type = 'saga'
    test_class = [0]
    save_images = False
    visualize = True
    visualize_mistakes = False
    image_suffix = '.png'
else:
    rand_seed = flags.seed
    dataset = flags.dataset
    purpose = flags.purpose
    result_type = flags.result_type
    test_class = [flags.detector_test_class]
    save_images = flags.save_images
    visualize = flags.visualize
    visualize_mistakes = flags.visualize_mistakes
    image_suffix = flags.image_suffix

if result_type == 'pc':
    params['Epochs'] = 100
torch.manual_seed(rand_seed)
params['purpose'] = purpose
data_dir, models_dir, logs_dir, results_dir, images_dir = get_directories(dataset=dataset)
classifier_dir = os.path.join(models_dir, 'classifiers')
detector_dir = os.path.join(models_dir, 'detectors')
if not os.path.exists(detector_dir):
    os.mkdir(detector_dir)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if purpose == 'train':
    log_file = get_file(logs_dir, 'train_detector_' + dataset + '_' + result_type + '_' + current_time + '.txt')
else:  # purpose == 'test
    log_file = get_file(logs_dir, 'test_detector_' + dataset + '_' + result_type + '_' + current_time + '.txt')
device = get_device(log_file=log_file)
save_model_name = 'DTR_' + result_type + '_' + str(test_class[0])

save_faces_file = os.path.join(data_dir, 'raw', 'mesh_faces.npy')
assert os.path.exists(save_faces_file), 'saved faces file was not found'
mesh_faces = torch.from_numpy(np.load(save_faces_file)).to(device)

results_file = get_results_path(dataset, results_dir, result_type)
log_string(log_file, 'result_type: {}\nresults file: {}'.format(result_type, results_file))
assert os.path.exists(results_file), 'attack results file was not found'
with open(results_file, 'rb') as handle:
    all_pair_dict_list = pickle.load(handle)

# ---------- Considering only the end-of-attack savings ----------
attack_params = get_attack_params(dataset)
pair_dict_list = [item for item in all_pair_dict_list if
                  (item['step'] == attack_params['N_attack_steps'] - 1)]

log_string(log_file, 'dataset: {}, result_type: {}, results_file:\n{}\nsave_model_name:\n{}'
           .format(dataset, result_type, results_file, save_model_name))

# ---------- Dividing to train/validation/test ----------
train_s_mesh_list = [item['s_mesh'] for item in pair_dict_list if item['s_label'] not in test_class]
tmp_s_mesh_list = [item['s_mesh'] for item in pair_dict_list if item['s_label'] in test_class]
val_num_shapes = int(len(tmp_s_mesh_list) / 2)
val_s_mesh_list = tmp_s_mesh_list[:val_num_shapes]
test_s_mesh_list = tmp_s_mesh_list[val_num_shapes:]

train_adv_mesh_list = [item['adv_mesh'] for item in pair_dict_list if item['s_label'] not in test_class]
tmp_adv_mesh_list = [item['adv_mesh'] for item in pair_dict_list if item['s_label'] in test_class]
val_adv_mesh_list = tmp_adv_mesh_list[:val_num_shapes]
test_adv_mesh_list = tmp_adv_mesh_list[val_num_shapes:]

# ---------- Converting to torch ----------
train_s_mesh_list = [torch.from_numpy(item).to(device) for item in train_s_mesh_list]
val_s_mesh_list = [torch.from_numpy(item).to(device) for item in val_s_mesh_list]
test_s_mesh_list = [torch.from_numpy(item).to(device) for item in test_s_mesh_list]
train_adv_mesh_list = [torch.from_numpy(item).to(device) for item in train_adv_mesh_list]
val_adv_mesh_list = [torch.from_numpy(item).to(device) for item in val_adv_mesh_list]
test_adv_mesh_list = [torch.from_numpy(item).to(device) for item in test_adv_mesh_list]

# ---------- Labeling ----------
train_s_label_list = [torch.tensor(0).to(device=device, dtype=torch.int64) for item in train_s_mesh_list]
val_s_label_list = [torch.tensor(0).to(device=device, dtype=torch.int64) for item in val_s_mesh_list]
test_s_label_list = [torch.tensor(0).to(device=device, dtype=torch.int64) for item in test_s_mesh_list]

train_adv_label_list = [torch.tensor(1).to(device=device, dtype=torch.int64) for item in train_adv_mesh_list]
val_adv_label_list = [torch.tensor(1).to(device=device, dtype=torch.int64) for item in val_adv_mesh_list]
test_adv_label_list = [torch.tensor(1).to(device=device, dtype=torch.int64) for item in test_adv_mesh_list]

input_shape = train_s_mesh_list[0].shape

if purpose == 'train':
    detector = get_detector(params, device, input_shape)
    detector_dir_train = os.path.join(detector_dir, 'train_' + current_time)
    if not os.path.exists(detector_dir_train):
        os.mkdir(detector_dir_train)
    train_meshes = train_s_mesh_list + train_adv_mesh_list
    train_labels = train_s_label_list + train_adv_label_list
    log_string(log_file, "Number of pairs in the train set: {}".format(len(train_meshes)))

    val_meshes = val_s_mesh_list + val_adv_mesh_list
    val_labels = val_s_label_list + val_adv_label_list
    log_string(log_file, "Number of pairs in the validation set: {}".format(len(val_meshes)))

    # ---------- Optimizer ----------
    detector_optimizer = get_detector_optimizer(detector.parameters(), params['L_rate'])

    # ---------- Creating the dataset ---------
    num_evals = attack_params['N_evals']
    num_evects = attack_params['N_evects']
    train_num_shapes_for_dataset = len(train_meshes)
    num_vertices = train_meshes[0].shape[0]
    train_meshes = torch.stack(train_meshes).to(device)
    train_labels = torch.stack(train_labels).to(device)
    train_dummy_alphas = torch.zeros((train_num_shapes_for_dataset, num_evects, 3)).to(device)
    dummy_evects = torch.zeros((num_vertices, num_evects)).to(device)
    train_data = Data(train_meshes, train_dummy_alphas, train_labels, mesh_faces, dummy_evects, device)

    val_num_shapes_for_dataset = len(val_meshes)
    val_meshes = torch.stack(val_meshes).to(device)
    val_labels = torch.stack(val_labels).to(device)
    val_dummy_alphas = torch.zeros((val_num_shapes_for_dataset, num_evects, 3)).to(device)
    val_data = Data(val_meshes, val_dummy_alphas, val_labels, mesh_faces, dummy_evects, device)

    # ---------- Training ----------
    log_string(log_file, 'start detector training...')
    train_loss_values, val_loss_values = \
        train(train_data=train_data, model=detector, optimizer=detector_optimizer,
              params=params, save_model_dir=detector_dir_train, model_name=save_model_name,
              log_file=log_file, save_every=10, val_data=val_data)

    save_detector_training_plot = os.path.join(images_dir, 'DTR_' + dataset + '_' + purpose + '_' +
                                               result_type + '_' + str(test_class[0]) + '_' +
                                               str(params['Epochs']) + image_suffix)
    if visualize:
        plt.figure('Loss vs. Epoch')
        plt.title('Loss vs. Epoch')
        plt.plot(train_loss_values, 'cyan', label='training loss')
        plt.plot(val_loss_values, 'blue', label='validation loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        if save_images:
            plt.savefig(save_detector_training_plot)
            plt.close()
        else:
            plt.show()
else:  # purpose == 'test'
    dtr_dir_name = ''  # fill the training name.
    detector = load_trained_detector(params, detector_dir, device, input_shape,
                                     result_type=result_type, test_class=test_class[0],
                                     dtr_dir_name=dtr_dir_name)
    meshes = test_s_mesh_list + test_adv_mesh_list
    labels = test_s_label_list + test_adv_label_list
    log_string(log_file, "number of pairs in the test set: {}".format(len(meshes)))

    vis_list = []
    accuracy, confusion, detector_confidence_label_0, detector_confidence_label_1 = \
        evaluate(meshes, labels, detector, faces=mesh_faces, visualize=visualize,
                 vis_list=vis_list, visualize_mistakes=visualize_mistakes, log_file=log_file)
    log_string(log_file, "detector test after training.\n\naccuracy:\n{}\n\nconfusion:\n{}"
               .format(accuracy, confusion))



