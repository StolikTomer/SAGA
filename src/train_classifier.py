import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
import datetime
import time
import matplotlib.pylab as plt
from utils import get_directories, get_file, log_string, get_device, visualize_mesh_and_pc, get_argument_parser
from utils_data import load_data
from utils_models import get_classifier_params, get_classifier_optimizer, get_classifier, load_trained_classifier


def validate(val_data, model, criterion):
    model.eval()
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    val_loss = 0.0
    for x, _, label in val_dataloader:
        y = label.squeeze() if len(label.shape) > 1 else label
        batch_size = x.shape[0]
        out = model(x)
        out = out.view(batch_size, -1)
        loss = criterion(out, y)
        val_loss += loss.item()

    val_loss = val_loss / len(val_dataloader)
    return val_loss


def train(train_data, model, optimizer, save_model_dir, params, model_name, log_file=None, save_every=100,
          val_data=None):
    loss_values = []
    val_loss_values = []
    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_data, batch_size=np.int(params['B_size']), shuffle=True)

    for epoch in range(params['Epochs']):
        start_epoch_time = time.time()
        log_string(log_file, "epoch " + str(epoch + 1) + " of " + str(params['Epochs']))
        model.train()
        epoch_loss = 0.0
        for x, _, label in train_dataloader:
            y = label.squeeze() if len(label.shape) > 1 else label
            batch_size = x.shape[0]
            optimizer.zero_grad()
            out = model(x)
            out = out.view(batch_size, -1)

            loss = criterion(out, y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(train_dataloader)
        loss_values.append(epoch_loss)

        if val_data is not None:
            val_epoch_loss = validate(val_data, model, criterion)
            val_loss_values.append(val_epoch_loss)
            log_string(log_file, "train epoch duration (seconds) {}, train loss: {}, val loss: {}".format(
                (time.time() - start_epoch_time), epoch_loss, val_epoch_loss))
        else:
            log_string(log_file,
                       "train epoch duration (seconds) {}, train loss: {}".format((time.time() - start_epoch_time),
                                                                                  epoch_loss))
        if (epoch + 1) % save_every == 0 or epoch == 0:
            log_string(log_file, "saving model in epoch: {}".format(epoch))
            torch.save(model.state_dict(), os.path.join(save_model_dir, model_name + '_' + str(epoch) + '.h5'))

    torch.save(model.state_dict(), os.path.join(save_model_dir, model_name + '_' + str(epoch) + '.h5'))
    return loss_values, val_loss_values


def evaluate(meshes, labels, classifier, src_labels=None, faces=None, visualize=False,
             vis_list=[], visualize_mistakes=False, log_file=None):
    assert (len(meshes) == len(labels)), 'meshes and labels should be of the same length'
    if src_labels is not None:
        assert (len(meshes) == len(src_labels)), 'meshes and source labels should be of the same length'
    assert not (visualize and faces is None), 'cannot visualize when faces is None'

    faces_np = faces if isinstance(faces, np.ndarray) else faces.detach().cpu().numpy()
    start_time_tot = time.time()
    classifier.eval()
    confusion = None
    src_predictions = 0
    detector_confidence_label_0 = []
    detector_confidence_label_1 = []
    for i in tqdm.trange(len(meshes)):
        start_time = time.time()
        x = meshes[i]
        y = labels[i]
        if src_labels is not None:
            src_y = src_labels[i]

        out: torch.Tensor = classifier(x)
        if confusion is None:
            num_classes = out.shape[-1]
            confusion = torch.zeros([num_classes, num_classes])

        _, prediction = out.max(dim=-1)
        target = int(y)
        if num_classes == 2:  # detector only
            softmax = torch.nn.Softmax(dim=-1)
            if target == 0:
                detector_confidence_label_0.append(softmax(out)[1].detach().cpu().numpy())
            else:
                detector_confidence_label_1.append(softmax(out)[1].detach().cpu().numpy())

        if src_labels is not None:
            source = int(src_y)
            assert target != source, "the target label cannot be the same as the source label"
        if visualize and i in vis_list:
            if target == prediction:
                log_string(log_file, "\nCORRECT prediction ! the true label and prediction are: {}"
                           .format(prediction))
            else:
                if (src_labels is not None) and (source == prediction):
                    log_string(log_file, '\nVERY WRONG prediction ! the SOURCE and the PREDICTION labels are the '
                                         'same. label: {} '.format(prediction))
                else:
                    log_string(log_file, '\nWRONG prediction ! the true label is: {}, while the prediction is: {}'
                               .format(target, prediction))
                if visualize_mistakes:
                    vertices_i_np = meshes[i].detach().cpu().numpy()
                    visualize_mesh_and_pc(vertices_i_np, faces_np,
                                          window_name="shape idx {}, true label: {}, prediction: {}"
                                          .format(i, target, prediction))
            vertices_i_np = meshes[i].detach().cpu().numpy()
            visualize_mesh_and_pc(vertices_i_np, faces_np,
                                  window_name="shape idx {}, true label: {}, prediction: {}"
                                  .format(i, target, prediction))
        if (src_labels is not None) and (source == prediction):
            src_predictions += 1
        confusion[target, prediction] += 1

        correct = torch.diag(confusion).sum()
        accuracy = correct / confusion.sum()
        if src_labels is not None:
            untargeted_accuracy = (confusion.sum() - src_predictions) / confusion.sum()
    if src_labels is not None:
        log_string(log_file, 'time for the complete evaluation (sec): {}, accuracy is: {}, untargeted accuracy is: {'
                             '}, number of evaluated shapes: {} '
                   .format((time.time() - start_time_tot), accuracy, untargeted_accuracy, len(meshes)))
        return accuracy, untargeted_accuracy, confusion
    else:
        log_string(log_file, 'time for the complete evaluation (sec): {}, accuracy is: {}, number of evaluated '
                             'shapes: {} '
                   .format((time.time() - start_time_tot), accuracy, len(meshes)))
        if len(detector_confidence_label_0) or len(detector_confidence_label_1):
            return accuracy, confusion, np.stack(detector_confidence_label_0), np.stack(detector_confidence_label_1)
        else:
            return accuracy, confusion


if __name__ == "__main__":
    debug_mode = False
    flags = get_argument_parser()

    cls_params = get_classifier_params()
    if debug_mode:
        seed = 1
        dataset = 'coma'
        purpose = 'val'
        visualize = True
        visualize_mistakes = False
        save_images = False
        image_suffix = '.png'
    else:
        seed = flags.seed
        dataset = flags.dataset
        purpose = flags.purpose
        visualize = flags.visualize
        visualize_mistakes = flags.visualize_mistakes
        save_images = flags.save_images
        image_suffix = flags.image_suffix

    cls_params['purpose'] = purpose
    # ---------- Configurations ----------
    torch.manual_seed(seed)
    data_dir, models_dir, logs_dir, _, images_dir = get_directories(dataset=dataset)
    classifier_dir = os.path.join(models_dir, 'classifiers')
    if not os.path.exists(classifier_dir):
        os.mkdir(classifier_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = get_file(logs_dir, 'train_classifier_' + dataset + '_' + current_time + '.txt')
    device = get_device(log_file=log_file)

    log_string(log_file, "dataset = {}, purpose = {}, seed = {}\nclassifier params:\n\n{}\n\n"
               .format(dataset, purpose, seed, cls_params))

    data = load_data(dataset=dataset, params=cls_params, data_dir=data_dir, device=device, log_file=log_file)

    if visualize:  # visualize the loaded dataset for debug
        vis_idx_list = []
        for vis_idx in vis_idx_list:
            vertices_i_np = data.get_pos(vis_idx).detach().cpu().numpy()
            faces_i_np = data.get_faces().detach().cpu().numpy()
            visualize_mesh_and_pc(vertices_i_np, faces_i_np,
                                  window_name="shape idx {}, label: {}".format(vis_idx, data.get_label(vis_idx)))

    if purpose == 'train':
        classifier = get_classifier(cls_params, device, dataset)
        classifier_dir_train = os.path.join(classifier_dir, 'train_' + current_time)
        if not os.path.exists(classifier_dir_train):
            os.mkdir(classifier_dir_train)
        # ---------- Optimizer ----------
        classifier_optimizer = get_classifier_optimizer(classifier.parameters(), cls_params['L_rate'])

        # ---------- Training ----------
        log_string(log_file, 'start classifier training...')
        loss_values, _ = train(train_data=data, model=classifier, optimizer=classifier_optimizer, params=cls_params,
                               save_model_dir=classifier_dir_train, model_name='CLS', log_file=log_file)

        save_classifier_training_plot = os.path.join(images_dir, 'CLS_train_' + dataset + '_' + purpose +
                                                     '_' + str(cls_params['Epochs']) + '_' + current_time + image_suffix)
        if visualize:
            plt.figure('Loss vs. Epoch')
            plt.title('Loss vs. Epoch')
            plt.plot(loss_values, 'cyan', label='training loss')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            if save_images:
                plt.savefig(save_classifier_training_plot, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    else:  # purpose is 'val':
        classifier = load_trained_classifier(cls_params, classifier_dir, device, dataset)
        meshes = [data.get_pos(x) for x in range(data.get_num_shapes())]
        labels = [data.get_label(x) for x in range(data.get_num_shapes())]
        faces = data.get_faces()
        vis_list = []

        accuracy, confusion = evaluate(meshes, labels, classifier, src_labels=None, faces=faces, visualize=visualize,
                                       vis_list=vis_list, visualize_mistakes=visualize_mistakes, log_file=log_file)
        log_string(log_file, "classifier evaluation after training.\n\naccuracy:\n{}\n\nconfusion:\n{}"
                   .format(accuracy, confusion))
