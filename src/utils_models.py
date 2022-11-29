import os
import torch
import torch.nn as nn
from classifier import SimplePointNet, PointNetClassifier, FcClassifier
from autoencoder import SpatialAE


def get_ae_params():
    params = {'reduced_memory_mode': 'none',
              'purpose': 'train',  # ('train' 'val' 'test')
              'N_units': 30,  # Latent space dimension
              'L_rate_AE': 1e-4,  # Learning rate
              'B_size': 16,  # Batch Size
              'Epochs': 2000,  # Number of epochs
              'enc_layers': [300, 200],  # Encoder Layers
              'dec_layers': [200],  # Decoder Layers
              'AE_activation': 'tanh',  # AE Activation function
              }
    return params


def get_classifier_params():
    params = {'reduced_memory_mode': 'none',
              'purpose': 'train',  # ('train' 'val' 'test')
              'L_rate': 1e-3,  # Learning rate
              'B_size': 6,  # Learning rate
              'Epochs': 1000,  # Number of epochs
              'Latent_Space': 128,
              'Conv_Output_Dim': 512,
              'Conv_Layer_Sizes': [32, 128, 256],
              'FC_Layer_Sizes': [512, 256, 128],
              'Transform_Pos': [0],
              }
    return params


def get_detector_params():
    params = {'reduced_memory_mode': 'none',
              'purpose': 'train',  # ('train' 'val' 'test')
              'L_rate': 1e-5,  # Learning rate
              'B_size': 6,  # Learning rate
              'Epochs': 200,  # Number of epochs
              'Pure_FC_Layers_Sizes': [300, 200],
              }
    return params


def get_ae_optimizer(parameters, lr):
    return torch.optim.Adam(parameters, lr=lr, eps=1e-07)


def get_classifier_optimizer(parameters, lr):
    return torch.optim.Adam(parameters, lr=lr, weight_decay=5e-4)


def get_detector_optimizer(parameters, lr):
    return torch.optim.Adam(parameters, lr=lr, weight_decay=5e-4)


def get_autoencoder(ae_params, input_shape):
    return SpatialAE(ae_params, input_shape)


def get_classifier(cls_params, device, dataset):
    num_classes = 12 if dataset == 'coma' else 5
    # Encoder
    ENC = SimplePointNet(latent_dimensionality=cls_params['Latent_Space'],
                         convolutional_output_dim=cls_params['Conv_Output_Dim'],
                         conv_layer_sizes=cls_params['Conv_Layer_Sizes'],
                         fc_layer_sizes=cls_params['FC_Layer_Sizes'],
                         transformer_positions=cls_params['Transform_Pos']).to(device)

    # Classifier
    CLA = nn.Sequential(nn.Linear(cls_params['Latent_Space'], 64), nn.ReLU(), nn.Linear(64, num_classes)).to(device)

    # Model
    model = PointNetClassifier(ENC, CLA, cls_params['Latent_Space']).to(device)

    return model


def get_detector(params, device, input_shape):
    num_classes = 2
    return FcClassifier(input_shape, num_classes, params).to(device)


def load_trained_ae(ae_params, ae_dir, device, input_shape, ae_dir_name='official'):
    load_dir = os.path.join(ae_dir, ae_dir_name)
    load_idx = '1999'
    load_path = os.path.join(load_dir, 'AE_' + str(load_idx) + '.h5')
    assert os.path.exists(load_path), "the autoencoder's weights path does not exit. path: {}" \
                                      .format(load_path)

    AE = get_autoencoder(ae_params, input_shape)
    AE.load_state_dict(torch.load(load_path))

    """
    Since our adversary input to the model is built from tensors that require grad,
    every operation in the forward pass will include at least one tensor that requires grad and thus 
    the method save_for_backwards will be used and gradients will be calculated.
    However, in order to keep the models' parameters fixed during the optimization process, 
    we avoid calculating their gradients (they are also not included in the optimizer's parameters).
    """
    for parameter in AE.parameters():
        parameter.requires_grad = False

    AE.eval()
    AE.to(device)

    return AE


def load_trained_classifier(cls_params, cls_dir, device, dataset, cls_dir_name='official'):
    load_dir = os.path.join(cls_dir, cls_dir_name)
    load_idx = 999
    load_path = os.path.join(load_dir, 'CLS_' + str(load_idx) + '.h5')
    assert os.path.exists(load_path), "the classifier's weights path does not exit. path: {}" \
                                      .format(load_path)

    classifier = get_classifier(cls_params, device, dataset)
    classifier.load_state_dict(torch.load(load_path))

    return classifier


def load_trained_detector(dtr_params, dtr_dir, device, input_shape,
                          result_type='saga', test_class=0, dtr_dir_name='official'):
    load_dir = os.path.join(dtr_dir, dtr_dir_name)
    load_idx = 199 if result_type == 'saga' else 99
    load_path = os.path.join(load_dir, 'DTR_' + result_type + '_' + str(test_class) + '_' + str(load_idx) + '.h5')
    assert os.path.exists(load_path), "the detector's weights path does not exist. path: {}".format(load_path)

    detector = get_detector(dtr_params, device, input_shape)
    detector.load_state_dict(torch.load(load_path))

    return detector
