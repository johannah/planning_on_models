# baseline learning vq-vae
# very strongly referenced vq-vae code from Ritesh Kumar from below link:
# https://github.com/ritheshkumar95/vq-vae-exps/blob/master/vq-vae/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from IPython import embed
from pixel_cnn import GatedPixelCNN
from ae_utils import discretized_mix_logistic_loss

class VQVAE_ENCODER(nn.Module):
    def __init__(self, num_clusters=512, encoder_output_size=32,
                 in_channels_size=1):
        super(VQVAE_ENCODER, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_size,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64,
                      out_channels=encoder_output_size,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(encoder_output_size),
            )
        ## vq embedding scheme
        self.embedding = nn.Embedding(num_clusters, encoder_output_size)
        # common scaling for embeddings - variance roughly scales with num_clusters
        self.embedding.weight.data.copy_(1./num_clusters *
                                torch.randn(num_clusters,encoder_output_size))
    def forward(self, x):
        # get continuous output directly from encoder
        z_e_x = self.encoder(x)
        # NCHW is the order in the encoder
        # (num, channels, height, width)
        N, C, H, W = z_e_x.size()
        # need NHWC instead of default NCHW for easier computations
        z_e_x_transposed = z_e_x.permute(0,2,3,1)
        # needs C,K
        emb = self.embedding.weight.transpose(0,1)
        # broadcast to determine distance from encoder output to clusters
        # NHWC -> NHWCK
        measure = z_e_x_transposed.unsqueeze(4) - emb[None, None, None]
        # square each element, then sum over channels
        dists = torch.pow(measure, 2).sum(-2)
        # pytorch gives real min and arg min - select argmin
        # this is the closest k for each sample - Equation 1
        # latents is array of integers
        latents = dists.min(-1)[1]

        # look up cluster centers
        z_q_x = self.embedding(latents.view(latents.size(0), -1))
        # back to NCHW (orig) - now cluster centers/class
        z_q_x = z_q_x.view(N, H, W, C).permute(0, 3, 1, 2)
        return z_e_x, z_q_x, latents

class VQVAE_PCNN_DECODER(nn.Module):
    def __init__(self, n_filters, n_layers, hsize, wsize, num_output_channels,
                 float_condition_size=None, spatial_condition_size=None, n_classes=None,
                 ):
        super(VQVAE_PCNN_DECODER, self).__init__()
        input_dim = 1
        last_layer_bias = 0.5
        self.num_output_channels = num_output_channels
        self.pcnn_decoder = GatedPixelCNN(input_dim=input_dim,
                                          dim=n_filters,
                                          n_layers=n_layers,
                                          n_classes=n_classes,
                                          float_condition_size=float_condition_size,
                                          spatial_condition_size=spatial_condition_size,
                                          last_layer_bias=last_layer_bias,
                                          hsize=hsize, wsize=wsize)
        if self.num_output_channels > 1:
            print('making output conv')
            # make channels for loss
            self.output_conv = nn.Conv2d(in_channels=1,
                      out_channels=self.num_output_channels,
                      kernel_size=1,
                      stride=1, padding=0)

    def forward(self, y, class_condition=None, float_condition=None, spatial_condition=None):
        yhat = self.pcnn_decoder(x=y, class_condition=class_condition,
                          float_condition=float_condition,
                          spatial_condition=spatial_condition
                          )
        # get number of channels needed for discretized mix loss
        if self.num_output_channels > 1:
            yhat = self.output_conv(yhat)
        return yhat

class VQVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(64*10*10, 1*80*80)

    def encode(self, x, class_condition=None):
        z_e_x, z_q_x, latents = self.encoder(x)
        bs,c,h,w = z_q_x.shape
        bs,yc,yh,yw = x[:,-1:,:,:].shape
        # turn z_q_x into same size as the y so we can add as input
        # seem to not get a gradient from the z_q_x ..... so combine with y and
        # use it that way. Needs more investigation
        spatial_condition = torch.autograd.Variable(z_q_x, requires_grad=True)
        scl = self.linear(spatial_condition.contiguous().view(bs,c*h*w)).contiguous().view(bs,yc,yh,yw)
        return z_e_x, z_q_x, latents, scl

    def forward(self, x, y, class_condition=None):
        z_e_x, z_q_x, latents = self.encoder(x)
        bs,c,h,w = z_q_x.shape
        bs,yc,yh,yw = y.shape
        # turn z_q_x into same size as the y so we can add as input
        # seem to not get a gradient from the z_q_x ..... so combine with y and
        # use it that way. Needs more investigation
        self.spatial_condition = torch.autograd.Variable(z_q_x, requires_grad=True)
        self.y = torch.autograd.Variable(y, requires_grad=True)
        self.scl = self.linear(self.spatial_condition.contiguous().view(bs,c*h*w)).contiguous().view(bs,yc,yh,yw)
        self.yin = self.scl+self.y
       # x_d =  self.decoder(y=self.yin, class_condition=class_condition, spatial_condition=self.spatial_condition)
        x_d =  self.decoder(y=self.yin, class_condition=class_condition)
        return x_d, z_e_x, z_q_x, latents, self.scl

def get_vqvae_loss(x_d, target, z_e_x, z_q_x, nr_logistic_mix, beta, device):
    loss_1 = discretized_mix_logistic_loss(x_d, target, nr_mix=nr_logistic_mix, DEVICE=device)
    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss_3 = beta*F.mse_loss(z_e_x, z_q_x.detach())
    return loss_1, loss_2, loss_3


