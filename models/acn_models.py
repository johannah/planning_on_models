import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.nn import functional as F
import torch
from IPython import embed
# fast vq from rithesh
from functions import vq, vq_st
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class ConvEncoder(nn.Module):
    def __init__(self, code_len, input_size=1, encoder_output_size=1000):
        super(ConvEncoder, self).__init__()
        self.code_len = code_len
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=2,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=code_len*2,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(code_len*2),
            nn.ReLU(True),
           )
        # found via experimentation
        self.eo = eo = encoder_output_size
        self.fc21 = nn.Linear(eo, code_len)
        self.fc22 = nn.Linear(eo, code_len)

    def encode(self, x):
        o = self.encoder(x)
        ol = o.view(o.shape[0], o.shape[1]*o.shape[2]*o.shape[3])
        return self.fc21(ol), self.fc22(ol)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            o = eps.mul(std).add_(mu)
            return o
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class ConvEncodeDecodeLarge(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1, encoder_output_size=1000, last_layer_bias=0.5):
        super(ConvEncodeDecodeLarge, self).__init__()
        self.code_len = code_len
        self.encoder_output_size = encoder_output_size
        # find reshape to match encoder --> eo is 4 with mnist (28,28)  and
        # code_len of 64
        self.eo = np.sqrt(encoder_output_size/(2*code_len))
        assert self.eo == int(self.eo)
        self.eo = int(self.eo)

        # architecture dependent
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=2,
                      stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16,
                      out_channels=code_len*2,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(code_len*2),
            nn.ReLU(True),
           )
        # found via experimentation
        n = 16
        self.fc21 = nn.Linear(encoder_output_size, code_len)
        self.fc22 = nn.Linear(encoder_output_size, code_len)
        self.fc3 = nn.Linear(code_len, encoder_output_size)
        self.out_layer = nn.ConvTranspose2d(in_channels=n,
                         out_channels=output_size,
                         kernel_size=1,
                         stride=1, padding=0)

        # set bias to 0.5 for sigmoid with bce - 0 when using dml
        self.out_layer.bias.data.fill_(last_layer_bias)

        self.decoder = nn.Sequential(
                # 4x4
                nn.ConvTranspose2d(in_channels=code_len*2,
                       out_channels=n,
                       kernel_size=1,
                       stride=1, padding=0),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                 # 4x4 -->  8x8
                 nn.ConvTranspose2d(in_channels=n,
                       out_channels=n,
                       kernel_size=4,
                       stride=2, padding=1),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                 # 8>14
                 nn.ConvTranspose2d(in_channels=n,
                         out_channels=n,
                         kernel_size=2,
                         stride=2, padding=1),
                 nn.BatchNorm2d(16),
                 nn.ReLU(True),
                 # 14->28
                 nn.ConvTranspose2d(in_channels=n,
                         out_channels=n,
                         kernel_size=2,
                         stride=2, padding=0),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                self.out_layer,
                )


    def encode(self, x):
        o = self.encoder(x)
        ol = o.view(o.shape[0], o.shape[1]*o.shape[2]*o.shape[3])
        return self.fc21(ol), self.fc22(ol)

    def decode(self, z):
        co = F.relu(self.fc3(z))
        col = co.view(co.shape[0], self.code_len*2, self.eo, self.eo)
        do = self.decoder(col)
        return do

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            o = eps.mul(std).add_(mu)
            return o
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

class ConvEncodeDecode(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1, encoder_output_size=1000, last_layer_bias=0.5):
        super(ConvEncodeDecode, self).__init__()
        self.code_len = code_len
        # find reshape to match encoder --> eo is 4 with mnist (28,28)  and
        # code_len of 64
        # eo should be 7 for mnist
        self.eo = np.sqrt(encoder_output_size/(2*code_len))
        assert self.eo == int(self.eo)
        self.eo = int(self.eo)
        # architecture dependent
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
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
                      out_channels=code_len*2,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(code_len*2),
            nn.ReLU(True),
           )
        self.fc21 = nn.Linear(encoder_output_size, code_len)
        self.fc22 = nn.Linear(encoder_output_size, code_len)
        self.fc3 = nn.Linear(code_len, encoder_output_size)

        self.out_layer = nn.ConvTranspose2d(in_channels=16,
                        out_channels=output_size,
                        kernel_size=4,
                        stride=2, padding=1)

        # set bias to 0.5 for sigmoid with bce - 0 when using dml
        self.out_layer.bias.data.fill_(last_layer_bias)

        self.decoder = nn.Sequential(
               nn.ConvTranspose2d(in_channels=code_len*2,
                      out_channels=32,
                      kernel_size=1,
                      stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=32,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                self.out_layer
                     )

    def encode(self, x):
        o = self.encoder(x)
        ol = o.view(o.shape[0], o.shape[1]*o.shape[2]*o.shape[3])
        return self.fc21(ol), self.fc22(ol)

    def decode(self, z):
        co = F.relu(self.fc3(z))
        col = co.view(co.shape[0], self.code_len*2, self.eo, self.eo)
        do = self.decoder(col)
        return do

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            o = eps.mul(std).add_(mu)
            return o
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

class Upsample(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(Upsample, self).__init__()
        """"
        test code to upsample forcibly downsampled image for spatial conditioning
        expects image of size bs,x,7,7 and will output bs,x,28,28
        """
        n = 16
        self.out_layer = nn.ConvTranspose2d(in_channels=n,
                        out_channels=output_size,
                        kernel_size=4,
                        stride=2, padding=1)

        self.upsample = nn.Sequential(
               nn.ConvTranspose2d(in_channels=input_size,
                      out_channels=n,
                      kernel_size=1,
                      stride=1, padding=0),
                nn.BatchNorm2d(n),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=n,
                      out_channels=n,
                      kernel_size=4,
                      stride=2, padding=1),
                nn.BatchNorm2d(n),
                nn.ReLU(True),
                self.out_layer
                     )

    def forward(self, x):
        return self.upsample(x)


class ConvEncodeDecodeLargeVQVAE(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       encoder_output_size=1000, last_layer_bias=0.5, num_clusters=512, num_z=32):
        super(ConvEncodeDecodeLargeVQVAE, self).__init__()
        self.code_len = code_len
        self.encoder_output_size = encoder_output_size
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.eo = np.sqrt(encoder_output_size/code_len)
        assert self.eo == int(self.eo)
        self.eo = int(self.eo)
        # code_len of 64
        n = 16
        # architecture dependent
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=n,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(True),
            nn.Conv2d(in_channels=n,
                      out_channels=n,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(True),
            nn.Conv2d(in_channels=n,
                      out_channels=n,
                      kernel_size=2,
                      stride=2, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(True),
            nn.Conv2d(in_channels=n,
                      out_channels=num_z,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(num_z),
            nn.ReLU(True),
           )

        self.fc21 = nn.Linear(encoder_output_size, code_len)
        self.fc22 = nn.Linear(encoder_output_size, code_len)
        self.fc3 = nn.Linear(code_len, encoder_output_size)
        self.conv_layers = nn.Sequential(
                               nn.ConvTranspose2d(in_channels=self.code_len,
                                  out_channels=num_z,
                                  kernel_size=1,
                                  stride=1, padding=0),
                               nn.BatchNorm2d(num_z),
                               nn.ReLU(True),
                               nn.ConvTranspose2d(in_channels=num_z,
                                  out_channels=num_z,
                                  kernel_size=1,
                                  stride=1, padding=0),
                               )

        self.out_layer = nn.ConvTranspose2d(in_channels=n,
                         out_channels=output_size,
                         kernel_size=1,
                         stride=1, padding=0)

        # set bias to 0.5 for sigmoid with bce - 0 when using dml
        self.out_layer.bias.data.fill_(last_layer_bias)
        ## vq embedding scheme
        #self.embedding = nn.Embedding(num_clusters, num_z)
        # common scaling for embeddings - variance roughly scales with num_clusters
        #self.embedding.weight.data.copy_(1./num_clusters * torch.randn(num_clusters, num_z))
        # from Rithesh
        self.codebook = VQEmbedding(num_clusters, num_z)

        self.decoder = nn.Sequential(
                # 4x4
                nn.ConvTranspose2d(in_channels=num_z,
                       out_channels=n,
                       kernel_size=1,
                       stride=1, padding=0),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                 # 4x4 -->  8x8
                 nn.ConvTranspose2d(in_channels=n,
                       out_channels=n,
                       kernel_size=4,
                       stride=2, padding=1),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                 # 8>14
                 nn.ConvTranspose2d(in_channels=n,
                         out_channels=n,
                         kernel_size=2,
                         stride=2, padding=1),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                 # 14->28
                 nn.ConvTranspose2d(in_channels=n,
                         out_channels=n,
                         kernel_size=2,
                         stride=2, padding=0),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                self.out_layer,
                )
        self.apply(weights_init)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            o = eps.mul(std).add_(mu)
            return o
        else:
            return mu

    def acn_encode(self, x):
        o = self.encoder(x)
        ol = o.view(o.shape[0], o.shape[1]*o.shape[2]*o.shape[3])
        mu = self.fc21(ol)
        logvar =self.fc22(ol)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def vq_encode(self, z):
        # convert z to image shape
        co = F.relu(self.fc3(z))
        co = co.view(z.shape[0], self.code_len, self.eo, self.eo)
        z_e_x = self.conv_layers(co)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_q_x

    def forward(self, x):
        z, mu, logvar = self.acn_encode(x)
        z_e_x, latents = self.vq_encode(z)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z, mu, logvar, z_e_x, z_q_x, latents

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar

# rithesh version
# https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/vqvae.py
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class VQVAEres(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=256,
                       last_layer_bias=0.5, num_clusters=512, num_z=32):

        super(VQVAEres, self).__init__()
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
            ResBlock(hidden_size),
            ResBlock(hidden_size),
            )
        self.codebook = VQEmbedding(num_clusters, hidden_size)
        self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def decode(self, latents):
        # TODO order/?
        z_q_x = self.codebook.embedding(latents).permute(0,3,1,2) # BDHW
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_q_x

    def forward(self, x):
        z_e_x, latents = self.encode(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, latents

class midACNVQVAEres(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                 num_clusters=512, num_z=32, num_actions=3, num_rewards=2, small=False):

        super(midACNVQVAEres, self).__init__()
        self.code_len = code_len
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 8
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.small = small
        fsos = first_stage_output_size = 128
        encoder_match_size = ems = hidden_size
        self.action_encoder = nn.Sequential(
                               nn.Conv2d(num_actions, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, ems, 1, 1, 0),
                               nn.ReLU(True),
                             )
        self.reward_encoder = nn.Sequential(
                               nn.Conv2d(num_rewards, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, ems, 1, 1, 0),
                               nn.ReLU(True),
                             )
        self.frame_encoder = nn.Sequential(
                               nn.Conv2d(input_size, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, fsos, 1, 1, 0),
                               nn.ReLU(True),
                             )

        if self.small:
            self.encoder = nn.Sequential(
                nn.Conv2d(fsos, hidden_size, 4, 2, 0),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
                nn.Conv2d(hidden_size, hidden_size, 2, 1, 0),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, 3, 1, 1, 0),
                # need to get small enough to have reasonable knn - this is
                # 3*8*8=192
                )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(fsos, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 0),
                nn.Conv2d(hidden_size, hidden_size, 2, 1, 0),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, 3, 1, 1, 0),
                # need to get small enough to have reasonable knn - this is
                # 3*8*8=192
                )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 1, 1, 0),
            nn.Conv2d(16, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            ResBlock(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            )
        self.codebook = VQEmbedding(num_clusters, hidden_size)
        if self.small:
            self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 2, 1, 0),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 1, 1, 0),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 0),
                  )
        else:
            self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 2, 1, 0),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 0),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def vq_encode(self, mu):
        z_e_x = self.conv_layers(mu)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def forward(self, frames):
        bs, c, h, w = frames.shape
        x = self.frame_encoder(frames)
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z, actions, rewards):
        z_e_x, latents = self.vq_encode(z)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st + self.action_encoder(actions) + self.reward_encoder(rewards))
        return x_tilde, z_e_x, z_q_x, latents


class fwdACNVQVAEres(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                 num_clusters=512, num_z=32, num_actions=3, num_rewards=2, small=False):

        super(fwdACNVQVAEres, self).__init__()
        self.code_len = code_len
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 8
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.small = small
        fsos = first_stage_output_size = 64
        self.action_encoder = nn.Sequential(
                               nn.Conv2d(num_actions, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, fsos, 1, 1, 0),
                               nn.ReLU(True),
                             )
        self.reward_encoder = nn.Sequential(
                               nn.Conv2d(num_rewards, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, fsos, 1, 1, 0),
                               nn.ReLU(True),
                             )
        self.frame_encoder = nn.Sequential(
                               nn.Conv2d(input_size, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, fsos, 1, 1, 0),
                               nn.ReLU(True),
                             )

        if self.small:
            self.encoder = nn.Sequential(
                nn.Conv2d(fsos*3, hidden_size, 4, 2, 0),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
                nn.Conv2d(hidden_size, hidden_size, 2, 1, 0),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, 3, 1, 1, 0),
                # need to get small enough to have reasonable knn - this is
                # 3*8*8=192
                )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(fsos*3, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 0),
                nn.Conv2d(hidden_size, hidden_size, 2, 1, 0),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, 3, 1, 1, 0),
                # need to get small enough to have reasonable knn - this is
                # 3*8*8=192
                )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 1, 1, 0),
            nn.Conv2d(16, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            ResBlock(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            )
        self.codebook = VQEmbedding(num_clusters, hidden_size)
        if self.small:
            self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 2, 1, 0),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 1, 1, 0),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 0),
                  )
        else:
            self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 2, 1, 0),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 0),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def vq_encode(self, mu):
        z_e_x = self.conv_layers(mu)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def forward(self, frames, actions, rewards):
        bs, c, h, w = frames.shape
        frames = self.frame_encoder(frames)
        actions = self.action_encoder(actions)
        rewards = self.reward_encoder(rewards)
        x = torch.cat((frames, actions, rewards), dim=1)
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z):
        z_e_x, latents = self.vq_encode(z)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, latents


class ACNVQVAEres(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                      num_clusters=512, num_z=32, small=False):

        super(ACNVQVAEres, self).__init__()
        self.code_len = code_len
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 8
        self.small = small
        fsos = first_stage_output_size = 64
        self.frame_encoder = nn.Sequential(
                               nn.Conv2d(input_size, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, fsos, 1, 1, 0),
                               nn.ReLU(True),
                             )

        if self.small:
            self.encoder = nn.Sequential(
                nn.Conv2d(fsos, hidden_size, 4, 2, 0),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
                nn.Conv2d(hidden_size, hidden_size, 2, 1, 0),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, 3, 1, 1, 0),
                # need to get small enough to have reasonable knn - this is
                # 3*8*8=192
                )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(fsos, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 0),
                nn.Conv2d(hidden_size, hidden_size, 2, 1, 0),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, 3, 1, 1, 0),
                # need to get small enough to have reasonable knn - this is
                # 3*8*8=192
                )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 1, 1, 0),
            nn.Conv2d(16, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            ResBlock(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            )
        self.codebook = VQEmbedding(num_clusters, hidden_size)
        if self.small:
            self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 2, 1, 0),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 1, 1, 0),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 0),
                  )
        else:
            self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 2, 1, 0),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 0),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def vq_encode(self, mu):
        z_e_x = self.conv_layers(mu)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def forward(self, frames):
        x = self.frame_encoder(frames)
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z):
        z_e_x, latents = self.vq_encode(z)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, latents

class ACNres(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1, encoder_output_size=1024,
                       hidden_size=256, use_decoder=True
                       ):

        super(ACNres, self).__init__()
        self.code_len = code_len
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.encoder_output_size = encoder_output_size
        self.eo = 7
        self.use_decoder = use_decoder

        self.encoder = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
            ResBlock(hidden_size),
            ResBlock(hidden_size),
            nn.Conv2d(hidden_size, 16, 1, 1, 0),
            nn.Conv2d(16, 4, 1, 1, 0),
            # need to get small enough to have reasonable knn - this is
            # 4*7*7=196
            )
        if self.use_decoder:
            self.decoder = nn.Sequential(
                  nn.Conv2d(4, 16, 1, 1, 0),
                  nn.Conv2d(16, hidden_size, 1, 1, 0),
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )
        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        return z, mu

class VQVAE(nn.Module):
     # reconstruction from this model is poor
    def __init__(self, input_size=1, output_size=1,
                       encoder_output_size=1000, last_layer_bias=0.5, num_clusters=512, num_z=32):
        super(VQVAE, self).__init__()
        self.encoder_output_size = encoder_output_size
        self.num_clusters = num_clusters
        self.num_z = num_z
        n = 16
        # architecture dependent
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=n,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(True),
            nn.Conv2d(in_channels=n,
                      out_channels=n,
                      kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(True),
            nn.Conv2d(in_channels=n,
                      out_channels=n,
                      kernel_size=2,
                      stride=2, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(True),
            nn.Conv2d(in_channels=n,
                      out_channels=num_z,
                      kernel_size=1,
                      stride=1, padding=0),
            nn.BatchNorm2d(num_z),
            nn.ReLU(True),
           )

        self.out_layer = nn.ConvTranspose2d(in_channels=n,
                         out_channels=output_size,
                         kernel_size=1,
                         stride=1, padding=0)

        # set bias to 0.5 for sigmoid with bce - 0 when using dml
        self.out_layer.bias.data.fill_(last_layer_bias)
        ## vq embedding scheme
        #self.embedding = nn.Embedding(num_clusters, num_z)
        # common scaling for embeddings - variance roughly scales with num_clusters
        #self.embedding.weight.data.copy_(1./num_clusters * torch.randn(num_clusters, num_z))
        # from Rithesh
        self.codebook = VQEmbedding(num_clusters, num_z)

        self.decoder = nn.Sequential(
                # 4x4
                nn.ConvTranspose2d(in_channels=num_z,
                       out_channels=n,
                       kernel_size=1,
                       stride=1, padding=0),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                 # 4x4 -->  8x8
                 nn.ConvTranspose2d(in_channels=n,
                       out_channels=n,
                       kernel_size=4,
                       stride=2, padding=1),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                 # 8>14
                 nn.ConvTranspose2d(in_channels=n,
                         out_channels=n,
                         kernel_size=2,
                         stride=2, padding=1),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                 # 14->28
                 nn.ConvTranspose2d(in_channels=n,
                         out_channels=n,
                         kernel_size=2,
                         stride=2, padding=0),
                 nn.BatchNorm2d(n),
                 nn.ReLU(True),
                self.out_layer,
                )
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_q_x

    def forward(self, x):
        z_e_x, latents = self.encode(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, latents

class tPTPriorNetwork(nn.Module):
    def __init__(self, size_training_set, code_length, n_hidden=512, k=5, random_seed=4543):
        super(tPTPriorNetwork, self).__init__()
        self.rdn = np.random.RandomState(random_seed)
        self.k = k
        self.size_training_set = size_training_set
        self.code_length = code_length
        self.input_layer = nn.Linear(code_length, n_hidden)
        self.skipin_to_2 = nn.Linear(n_hidden, n_hidden)
        self.skipin_to_3 = nn.Linear(n_hidden, n_hidden)
        self.skip1_to_out = nn.Linear(n_hidden, n_hidden)
        self.skip2_to_out = nn.Linear(n_hidden, n_hidden)
        self.h1 = nn.Linear(n_hidden, n_hidden)
        self.h2 = nn.Linear(n_hidden, n_hidden)
        self.h3 = nn.Linear(n_hidden, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, self.code_length)
        self.fc_s = nn.Linear(n_hidden, self.code_length)

        # needs to be a param so that we can load
        self.codes = nn.Parameter(torch.FloatTensor(self.rdn.standard_normal((self.size_training_set, self.code_length))), requires_grad=False)
        batch_size = 64
        n_neighbors = 5
        self.neighbors = torch.LongTensor((batch_size, n_neighbors))
        self.distances = torch.FloatTensor((batch_size, n_neighbors))
        self.batch_indexer = torch.LongTensor(torch.arange(batch_size))

    def update_codebook(self, indexes, values):
        self.codes[indexes] = values

    def kneighbors(self, test, n_neighbors):
        with torch.no_grad():
            device = test.device
            bs = test.shape[0]
            return_size = (bs,n_neighbors)
            # dont recreate unless necessary
            if self.neighbors.shape != return_size:
                print('updating prior sizes')
                self.neighbors = torch.LongTensor(torch.zeros(return_size, dtype=torch.int64))
                self.distances = torch.zeros(return_size)
                self.batch_indexer = torch.LongTensor(torch.arange(bs))
            if device != self.codes.device:
                print('transferring prior to %s'%device)
                self.neighbors = self.neighbors.to(device)
                self.distances = self.distances.to(device)
                self.codes = self.codes.to(device)

            for bidx in range(test.shape[0]):
                dists = torch.norm(self.codes-test[bidx], dim=1)
                self.distances[bidx], self.neighbors[bidx] = dists.topk(n_neighbors, largest=False)
                del dists
        #print('kn', bidx, torch.cuda.memory_allocated(device=None))
        return self.distances.detach(), self.neighbors.detach()

    def batch_pick_close_neighbor(self, codes):
        '''
        :code latent activation of training
        '''
        neighbor_distances, neighbor_indexes = self.kneighbors(codes, n_neighbors=self.k)
        bsize = neighbor_indexes.shape[0]
        if self.training:
            # randomly choose neighbor index from top k
            #print('************ training - random neighbor')
            chosen_neighbor_index = torch.LongTensor(self.rdn.randint(0,neighbor_indexes.shape[1],size=bsize))
        else:
            chosen_neighbor_index = torch.LongTensor(torch.zeros(bsize, dtype=torch.int64))
        return self.codes[neighbor_indexes[self.batch_indexer, chosen_neighbor_index]]

    def forward(self, codes):
        previous_codes = self.batch_pick_close_neighbor(codes)
        return self.encode(previous_codes)

    def encode(self, prev_code):
        """
        The prior network was an
        MLP with three hidden layers each containing 512 tanh
        units
        - and skip connections from the input to all hidden
        layers and
        - all hiddens to the output layer.
        """
        i = torch.tanh(self.input_layer(prev_code))
        # input goes through first hidden layer
        _h1 = torch.tanh(self.h1(i))

        # make a skip connection for h layers 2 and 3
        _s2 = torch.tanh(self.skipin_to_2(_h1))
        _s3 = torch.tanh(self.skipin_to_3(_h1))

        # h layer 2 takes in the output from the first hidden layer and the skip
        # connection
        _h2 = torch.tanh(self.h2(_h1+_s2))

        # take skip connection from h1 and h2 for output
        _o1 = torch.tanh(self.skip1_to_out(_h1))
        _o2 = torch.tanh(self.skip2_to_out(_h2))
        # h layer 3 takes skip connection from prev layer and skip connection
        # from nput
        _o3 = torch.tanh(self.h3(_h2+_s3))

        out = _o1+_o2+_o3
        mu = self.fc_mu(out)
        logstd = self.fc_s(out)
        return mu, logstd


class PTPriorNetwork(nn.Module):
    def __init__(self, size_training_set, code_length, n_hidden=512, k=5, random_seed=4543):
        super(PTPriorNetwork, self).__init__()
        self.rdn = np.random.RandomState(random_seed)
        self.k = k
        self.size_training_set = size_training_set
        self.code_length = code_length
        self.fc1 = nn.Linear(self.code_length, n_hidden)
        self.fc2_u = nn.Linear(n_hidden, self.code_length)
        self.fc2_s = nn.Linear(n_hidden, self.code_length)
        #self.codes = nn.Parameter(torch.FloatTensor(self.rdn.standard_normal((self.size_training_set, self.code_length))))
        self.codes = torch.FloatTensor(self.rdn.standard_normal((self.size_training_set, self.code_length)))
        batch_size = 64
        n_neighbors = 5
        self.neighbors = torch.LongTensor((batch_size, n_neighbors))
        self.distances = torch.FloatTensor((batch_size, n_neighbors))
        self.batch_indexer = torch.LongTensor(torch.arange(batch_size))

    def update_codebook(self, indexes, values):
        self.codes[indexes] = values

    def kneighbors(self, test, n_neighbors):
        with torch.no_grad():
            device = test.device
            bs = test.shape[0]
            return_size = (bs,n_neighbors)
            # dont recreate unless necessary
            if self.neighbors.shape != return_size:
                print('updating prior sizes')
                self.neighbors = torch.LongTensor(torch.zeros(return_size, dtype=torch.int64))
                self.distances = torch.zeros(return_size)
                self.batch_indexer = torch.LongTensor(torch.arange(bs))
            if device != self.codes.device:
                print('transferring prior to %s'%device)
                self.neighbors = self.neighbors.to(device)
                self.distances = self.distances.to(device)
                self.codes = self.codes.to(device)

            for bidx in range(test.shape[0]):
                dists = torch.norm(self.codes-test[bidx], dim=1)
                self.distances[bidx], self.neighbors[bidx] = dists.topk(n_neighbors, largest=False)
                del dists
        #print('kn', bidx, torch.cuda.memory_allocated(device=None))
        return self.distances.detach(), self.neighbors.detach()

    def batch_pick_close_neighbor(self, codes):
        '''
        :code latent activation of training
        '''
        neighbor_distances, neighbor_indexes = self.kneighbors(codes, n_neighbors=self.k)
        bsize = neighbor_indexes.shape[0]
        if self.training:
            # randomly choose neighbor index from top k
            #print('************ training - random neighbor')
            chosen_neighbor_index = torch.LongTensor(self.rdn.randint(0,neighbor_indexes.shape[1],size=bsize))
        else:
            chosen_neighbor_index = torch.LongTensor(torch.zeros(bsize, dtype=torch.int64))
        return self.codes[neighbor_indexes[self.batch_indexer, chosen_neighbor_index]]

    def forward(self, codes):
        previous_codes = self.batch_pick_close_neighbor(codes)
        return self.encode(previous_codes)

    def encode(self, prev_code):
        h1 = F.relu(self.fc1(prev_code))
        mu = self.fc2_u(h1)
        logstd = self.fc2_s(h1)
        # logstd is trained with noise added to the the prev_code so it does not
        # perform well when the prev_code is passed a prev_code with no noise
        # (ie .eval() on the encoder model)
        return mu, logstd

#class PriorNetwork(nn.Module):
#    def __init__(self, size_training_set, code_length, n_hidden=512, k=5, random_seed=4543):
#        super(PriorNetwork, self).__init__()
#        self.rdn = np.random.RandomState(random_seed)
#        self.k = k
#        self.size_training_set = size_training_set
#        self.code_length = code_length
#        self.fc1 = nn.Linear(self.code_length, n_hidden)
#        self.fc2_u = nn.Linear(n_hidden, self.code_length)
#        self.fc2_s = nn.Linear(n_hidden, self.code_length)
#
#        self.knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
#        # codes are initialized randomly - Alg 1: initialize C: c(x)~N(0,1)
#        #
#        codes = self.rdn.standard_normal((self.size_training_set, self.code_length))
#        self.fit_knn(codes)
#
#    def fit_knn(self, codes):
#        ''' will reset the knn  given an nd array
#        '''
#        self.codes = codes
#        y = np.zeros((len(self.codes)))
#        self.knn.fit(self.codes, y)
#
#    def batch_pick_close_neighbor(self, codes):
#        '''
#        :code latent activation of training example as np
#        '''
#        neighbor_distances, neighbor_indexes = self.knn.kneighbors(codes, n_neighbors=self.k, return_distance=True)
#        bsize = neighbor_indexes.shape[0]
#        if self.training:
#            # randomly choose neighbor index from top k
#            chosen_neighbor_index = self.rdn.randint(0,neighbor_indexes.shape[1],size=bsize)
#        else:
#            chosen_neighbor_index = np.zeros((bsize), dtype=np.int)
#        return self.codes[neighbor_indexes[np.arange(bsize), chosen_neighbor_index]]
#
#    def forward(self, codes):
#        device = codes.device
#        np_codes = codes.cpu().detach().numpy()
#        previous_codes = self.batch_pick_close_neighbor(np_codes)
#        previous_codes = torch.FloatTensor(previous_codes).to(device)
#        return self.encode(previous_codes)
#
#    #def encode(self, prev_code):
#    #    h1 = F.relu(self.fc1(prev_code))
#    #    mu = self.fc2_u(h1)
#    #    return mu
#
#    def encode(self, prev_code):
#        h1 = F.relu(self.fc1(prev_code))
#        mu = self.fc2_u(h1)
#        logstd = self.fc2_s(h1)
#        return mu, logstd


