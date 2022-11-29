import torch
from torch import nn
from torch.nn import functional as F


class SimplePointNet(nn.Module):
    '''
    Simplified PointNet, without embedding transformer matrices.
    Akin to the method in Achlioptas et al, Learning Representations and
    Generative Models for 3D Point Clouds.

    E.g.
    s = SimplePointNet(100, 200, (25,50,100), (150,120))
    // Goes: 3 -> 25 -> 50 -> 100 -> 200 -> 150 -> 120 -> 100
    '''

    def __init__(self,
                 latent_dimensionality: int,
                 convolutional_output_dim: int,
                 conv_layer_sizes,
                 fc_layer_sizes,
                 transformer_positions,
                 end_in_batchnorm=False):

        super(SimplePointNet, self).__init__()
        self.LD = latent_dimensionality
        self.CD = convolutional_output_dim
        self.transformer_positions = transformer_positions
        self.num_transformers = len(self.transformer_positions)

        assert self.CD % 2 == 0, "Input Conv dim must be even"

        # Basic order #
        # B x N x 3 --Conv_layers--> B x C --Fc_layers--> B x L
        # We divide the output by two in the conv layers because we are using
        # both average and max pooling, which will be concatenated.
        self._conv_sizes = [3] + [k for k in conv_layer_sizes] + [self.CD]  # //2
        self._fc_sizes = [self.CD] + [k for k in fc_layer_sizes]

        ### Convolutional Layers ###
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self._conv_sizes[i], self._conv_sizes[i + 1], 1),
                nn.BatchNorm1d(self._conv_sizes[i + 1]),
                nn.ReLU()
            )
            for i in range(len(self._conv_sizes) - 1)
        ])

        ### Transformers ###
        # These are run and applied to the input after the corresponding convolutional
        # layer is run. Note that they never change the feature size (or indeed the
        # tensor shape in general).
        # E.g. if 0 is given in the positions, a 3x3 matrix set will be applied.
        self.transformers = nn.ModuleList([
            SimpleTransformer(self._conv_sizes[jj])
            for jj in self.transformer_positions
        ])

        ### Fully Connected Layers ###
        self.fc_layers = nn.ModuleList([
                                           nn.Sequential(
                                               nn.Linear(self._fc_sizes[i], self._fc_sizes[i + 1]),
                                               # nn.BatchNorm1d(self._fc_sizes[i+1]),
                                               nn.ReLU()
                                           )
                                           for i in range(len(self._fc_sizes) - 1)]
                                       +
                                       ([nn.Linear(self._fc_sizes[-1], self.LD), nn.BatchNorm1d(self.LD)]
                                        if end_in_batchnorm else
                                        [nn.Linear(self._fc_sizes[-1], self.LD)])
                                       )

    def move_eye(self, device):
        for t in self.transformers: t.move_eye(device)

    def forward(self, pos):
        '''
        Input: B x N x 3 point clouds (non-permuted)
        Output: B x LD embedded shapes
        '''
        P = pos
        num_in_dim = len(P.shape)
        if num_in_dim == 2:
            P = P[None, ...]

        P = P.permute(0, 2, 1)
        assert P.shape[1] == 3, "Unexpected shape"

        # Now P is B x 3 x N
        for i, layer in enumerate(self.conv_layers):
            if i in self.transformer_positions:
                T = self.transformers[self.transformer_positions.index(i)](P)
                P = layer(torch.bmm(T, P))
            else:
                P = layer(P)
        # Pool over the number of points.
        # i.e. P: B x C_D x N --Pool--> B x C_D x 1
        # Then, P: B x C_D x 1 --> B x C_D (after squeeze)
        # Note: F.max_pool1d(input, kernel_size)
        P = F.max_pool1d(P, P.shape[2]).squeeze(2)

        for j, layer in enumerate(self.fc_layers): P = layer(P)

        if num_in_dim == 2:  P = P[0]

        return P


# complete classifier, using encoder layers and classification layers
class PointNetClassifier(nn.Module):
    def __init__(self, enc, cla, LATENT_SPACE):
        super().__init__()
        self.cla = cla
        self.enc = enc
        self.LATENT_SPACE = LATENT_SPACE

    def forward(self, pos):
        lsp = self.enc(pos)[..., :self.LATENT_SPACE]
        return self.cla(lsp)


class SimpleTransformer(nn.Module):

    def __init__(self,
                 input_dimensionality,
                 convolutional_dimensions=(64, 64),
                 fc_dimensions=(64, 32)):

        super(SimpleTransformer, self).__init__()

        # Setting network dimensions
        self.input_feature_len = input_dimensionality
        self.conv_dims = [self.input_feature_len] + [a for a in convolutional_dimensions]
        self.fc_dims = [f for f in fc_dimensions]

        ### Convolutional Layers ###
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.conv_dims[i], self.conv_dims[i + 1], 1),
                nn.BatchNorm1d(self.conv_dims[i + 1]),
                nn.ReLU()
            )
            for i in range(len(self.conv_dims) - 1)
        ])

        ### Fully Connected Layers ###
        self.fc_layers = nn.ModuleList([
                                           nn.Sequential(
                                               nn.Linear(self.fc_dims[i], self.fc_dims[i + 1]),
                                               nn.ReLU()
                                           ) for i in range(len(self.fc_dims) - 1)]
                                       + [nn.Linear(self.fc_dims[-1], self.input_feature_len ** 2)]
                                       )

        ### Identity matrix added to the transformer at the end ###
        self.eye = torch.eye(self.input_feature_len)

    def forward(self, x):
        '''
        Input: B x F x N, e.g. F = 3 at the beginning
            i.e. expects a permuted point cloud batch
        Output: B x F x F set of transformation matrices
        '''
        SF = x.shape[1]  # Size of the features per point
        # assert SF == self.input_feature_len, "Untenable feature len"

        # Convolutional layers
        for i, layer in enumerate(self.conv_layers): x = layer(x)
        # Max pooling
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)
        # Fully connected layers
        for j, layer in enumerate(self.fc_layers): x = layer(x)
        # Fold into list of matrices
        # x = x.view(-1, SF, SF) + self.eye
        x = x.view(-1, SF, SF) + self.eye.to(x.device)
        # x += self.eye.to(device)
        return x

    def move_eye(self, device):
        self.eye = self.eye.to(device)


def pool(x: torch.Tensor, downscale_mat: torch.sparse.FloatTensor):
    return torch.sparse.mm(downscale_mat, x)


class FcClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, classifier_params):
        #  Input: bs x N x 3
        #  Output: bs x num_classes
        super(FcClassifier, self).__init__()
        self.classifier_params = classifier_params
        self._fc_sizes = [input_shape[0] * input_shape[1]] + classifier_params['Pure_FC_Layers_Sizes']
        self.fc_layers = nn.ModuleList([
                                           nn.Sequential(
                                               nn.Linear(self._fc_sizes[i], self._fc_sizes[i + 1]),
                                               # nn.BatchNorm1d(self._fc_sizes[i+1]),
                                               nn.ReLU()
                                           )
                                           for i in range(len(self._fc_sizes) - 1)]
                                       +
                                       [nn.Linear(self._fc_sizes[-1], num_classes)]
                                       )

    def forward(self, x):
        z = x
        num_in_dim = len(z.shape)
        if num_in_dim == 2:
            z = z[None, ...]
        assert len(z.shape) == 3, "Unexpected shape"

        bs, _, _ = z.size()  # bs = batch size
        z = z.view(bs, -1)  # flatten
        for layer in self.fc_layers:
            z = layer(z)
        if num_in_dim == 2:
            z = z[0]
        return z
