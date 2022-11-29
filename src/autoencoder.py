import numpy as np
import torch
import torch.nn as nn


class SpatialAE(nn.Module):
    def __init__(self, ae_params, input_shape):
        super(SpatialAE, self).__init__()
        self.ae_params = ae_params
        self.input_shape = input_shape
        self.E_input_shape = self.input_shape[0] * self.input_shape[1]
        self.latent_size = np.int64(self.ae_params['N_units'])
        self.E_lins_list = self.ae_params['enc_layers']
        self.D_lins_list = self.ae_params['dec_layers']
        self.E_lins_last_idx = len(self.E_lins_list) - 1
        self.D_lins_last_idx = len(self.D_lins_list) - 1
        self.ae_activation = self.ae_params['AE_activation']

        # Encoder - enc
        self.E_lins = nn.ModuleList([self.get_initialized_linear_layer(self.E_input_shape, self.E_lins_list[0])])
        for i in np.arange(0, self.E_lins_last_idx):
            self.E_lins.append(self.get_initialized_linear_layer(self.E_lins_list[i], self.E_lins_list[i + 1]))
        self.E_lins.append(self.get_initialized_linear_layer(self.E_lins_list[self.E_lins_last_idx], self.latent_size))

        # Decoder - dec
        self.D_lins = nn.ModuleList([self.get_initialized_linear_layer(self.latent_size, self.D_lins_list[0])])
        for i in np.arange(0, self.D_lins_last_idx):
            self.D_lins.append(self.get_initialized_linear_layer(self.D_lins_list[i], self.D_lins_list[i + 1]))
        self.D_last = self.get_initialized_linear_layer(self.D_lins_list[self.D_lins_last_idx], self.E_input_shape)

    @staticmethod
    def get_initialized_linear_layer(input_size, output_size):
        linear = nn.Linear(input_size, output_size)
        torch.nn.init.xavier_uniform_(linear.weight)
        linear.bias.data.fill_(0)
        return linear

    def get_regularized_layers(self):
        reg_layers = []
        for i in range(len(self.E_lins)):
            reg_layers.append(self.E_lins[i])
        for i in range(len(self.D_lins)):
            reg_layers.append(self.D_lins[i])
        return reg_layers

    def get_un_regularized_layers(self):
        unreg_layers = [self.D_last]
        return unreg_layers

    def forward(self, x):
        z_from_x = self.E_net(x)
        y = self.D_net(z_from_x)
        return y, z_from_x

    def E_net(self, x):
        z = x
        z = z.contiguous()
        bs, _, _ = z.size()  # bs = batch size
        z = z.view(bs, -1)  # flatten
        for i in range(len(self.E_lins)):
            z = self.E_lins[i](z)
            if self.ae_activation == 'tanh':
                z = torch.tanh(z)
            else:
                raise NotImplementedError
        return z

    def D_net(self, z):
        bs, _ = z.size()
        y = z
        for i in range(len(self.D_lins)):
            y = self.D_lins[i](y)
            if self.ae_activation == 'tanh':
                y = torch.tanh(y)
            else:
                raise NotImplementedError
        y = self.D_last(y)
        out_shape = [bs, self.input_shape[0], self.input_shape[1]]
        y = y.view(out_shape)
        return y
