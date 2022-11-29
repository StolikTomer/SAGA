import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import datetime
import matplotlib.pylab as plt
from utils import log_string, get_file, get_device, get_directories, get_argument_parser
from utils_data import load_data
from utils_models import get_autoencoder, get_ae_params, get_ae_optimizer

flags = get_argument_parser()

dataset = flags.dataset
rand_seed = flags.seed
visualize = flags.visualize
save_ae_every = flags.save_ae_every
save_images = flags.save_images
image_suffix = flags.image_suffix

torch.manual_seed(rand_seed)
ae_params = get_ae_params()
data_dir, models_dir, logs_dir, _, images_dir = get_directories(dataset=dataset)
autoencoder_dir = os.path.join(models_dir, 'autoencoders')
if not os.path.exists(autoencoder_dir):
    os.mkdir(autoencoder_dir)

# ---------- Configurations ----------
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
models_dir_save = os.path.join(autoencoder_dir, 'trained_' + current_time)
if not os.path.exists(models_dir_save):
    os.mkdir(models_dir_save)
save_ae_training_image = os.path.join(images_dir, dataset + '_ae_training' + current_time + image_suffix)
log_file = get_file(logs_dir, 'train_ae_' + dataset + '_' + current_time + '.txt')
log_string(log_file, "seed = {}".format(rand_seed))
log_string(log_file, 'models_dir_save: {}'.format(models_dir_save))
log_string(log_file, "ae params:\n\n{}\n\n".format(ae_params))
device = get_device(log_file=log_file)

# ---------- Load data ----------
data_struct = load_data(dataset=dataset, params=ae_params, data_dir=data_dir, device=device, log_file=log_file)
num_shapes = data_struct.get_num_shapes()
num_vertices = data_struct.get_num_vertices()
num_dims = data_struct.get_pos(0).shape[-1]

# Build Models
input_shape = [num_vertices, num_dims]
AE = get_autoencoder(ae_params, input_shape)
AE.train()
AE = AE.to(device)

# DataLoader
dataloader = DataLoader(data_struct, batch_size=np.int(ae_params['B_size']), shuffle=True)
generator_optimizer = get_ae_optimizer(AE.parameters(), ae_params['L_rate_AE'])

log_string(log_file, 'Start training...')
epoch_loss_ae_list = []
epoch_loss_total_list = []
for epoch in range(0, ae_params['Epochs']):
    start_epoch_time = time.time()
    batch_loss_ae = []
    batch_loss_total = []
    for vertices_batch, _, _ in dataloader:
        g_vertices, latent = AE(vertices_batch)
        loss_ae = torch.sum(torch.mean(torch.square(vertices_batch - g_vertices), dim=-1))

        loss_reg = torch.tensor(0.).to(device)
        for layer in AE.get_regularized_layers():
            loss_reg += torch.norm(layer.weight)
        loss_reg = loss_reg * 0.01

        loss_total = loss_ae + loss_reg

        generator_optimizer.zero_grad()
        loss_total.backward()
        generator_optimizer.step()
        batch_loss_ae.append(loss_ae.item())
        batch_loss_total.append(loss_total.item())

    epoch_loss_ae = np.mean(batch_loss_ae)
    epoch_loss_total = np.mean(batch_loss_total)
    epoch_loss_ae_list.append(epoch_loss_ae)
    epoch_loss_total_list.append(epoch_loss_total)

    end_epoch_time = time.time()
    epoch_duration = (end_epoch_time - start_epoch_time) / 60
    template = 'Epoch {}, out of {}, Duration (minutes) {}, Loss_ae: {}, loss_tot: {}'
    log_string(log_file, template.format(epoch + 1,
                                         ae_params['Epochs'],
                                         epoch_duration,
                                         epoch_loss_ae,
                                         epoch_loss_total))

    if (epoch + 1) % save_ae_every == 0 or epoch == 0:
        log_string(log_file, "Saving models in epoch: {}".format(epoch))
        torch.save(AE.state_dict(), os.path.join(models_dir_save, 'AE_' + str(epoch) + '.h5'))

plt.figure('Loss vs. Epoch')
plt.title('Loss vs. Epoch')
plt.plot(epoch_loss_ae_list, 'cyan', label='loss_ae')
plt.plot(epoch_loss_ae_list, 'red', label='loss_total')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
if save_images:
    plt.savefig(save_ae_training_image, bbox_inches='tight')
    plt.close()
else:
    plt.show()

log_string(log_file, "Saving models after training is done. epoch: {}".format(epoch))
torch.save(AE.state_dict(), os.path.join(models_dir_save, 'AE_' + str(epoch) + '.h5'))

log_string(log_file, 'End training')
log_file.close()
