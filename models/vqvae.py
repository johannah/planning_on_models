# baseline learning vq-vae
# very strongly referenced vq-vae code from Ritesh Kumar from below link:
# https://github.com/ritheshkumar95/vq-vae-exps/blob/master/vq-vae/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from IPython import embed

class VQVAE(nn.Module):
    def __init__(self, num_clusters=512, encoder_output_size=32,
                 in_channels_size=1, num_output_mixtures=30,
                 n_actions=0, int_reward=0, reward_value=0):
        super(VQVAE, self).__init__()
        # the encoder_output_size is the size of the vector that is compressed
        # with vector quantization. if it is too large, vector quantization
        # becomes more difficult. if it is too small, then the conv net has less
        # capacity.
        # 64 - the network seems to train fairly well in only one epoch -
        # 16 - the network was able to perform nearly perfectly after 100 epochs
        # the compression factor can be thought of as follows for an input space
        # of 40x40x1 and z output of 10x10x9 (512 = 2**9 = 9 bits)
        # (40x40x1x8)/(10x10x9) = 12800/900 = 14.22

        self.n_actions = n_actions
        self.int_reward = int_reward
        self.reward_value = reward_value
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

        self.decoder = nn.Sequential(
                nn.Conv2d(in_channels=encoder_output_size,
                          out_channels=64,
                          kernel_size=1,
                          stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # applies a 2d transposed convolution operator over input image
                # composed of several input planes. Can be seen as gradient of Conv2d
                # with respsct to its input. also known as fractionally-strided conv.
                nn.ConvTranspose2d(in_channels=64,
                      out_channels=32,
                      kernel_size=4,
                      stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=32,
                      out_channels=16,
                      kernel_size=4,
                      stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=16,
                        out_channels=num_output_mixtures,
                        kernel_size=4,
                        stride=2, padding=1),
                #nn.Sigmoid()
                )
        # size of the output of the latent space - bs,encoder_output_size,vq_space_dim,vq_space_dim
        vq_space_dim = 10
        if self.n_actions > 0:
            print("predicting actions", self.n_actions)
            self.action_conv = nn.Sequential(
                                    nn.Conv2d(in_channels=encoder_output_size,
                                       out_channels=encoder_output_size,
                                       kernel_size=3, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=encoder_output_size,
                                                        out_channels=encoder_output_size,
                                                        kernel_size=3, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=encoder_output_size,
                                                        out_channels=self.n_actions,
                                                        kernel_size=vq_space_dim, padding=0),
                                 )
        if self.int_reward > 0:
            # reward should be between 0 and int_reward
            print("predicting an int reward", self.int_reward)
            self.int_reward_conv = nn.Sequential(
                                    nn.Conv2d(in_channels=encoder_output_size,
                                       out_channels=encoder_output_size,
                                       kernel_size=3, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=encoder_output_size,
                                                        out_channels=encoder_output_size,
                                                        kernel_size=3, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=encoder_output_size,
                                                        out_channels=self.int_reward,
                                                        kernel_size=vq_space_dim, padding=0),
                                 )
        if self.reward_value > 0:
            print("predicting reward value")
            # reward should be between 0 and int_reward
            self.reward_val_conv = nn.Sequential(
                                    nn.Conv2d(in_channels=encoder_output_size,
                                       out_channels=encoder_output_size,
                                       kernel_size=3, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=encoder_output_size,
                                                        out_channels=encoder_output_size,
                                                        kernel_size=3, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(in_channels=encoder_output_size,
                                                        out_channels=1,
                                                        kernel_size=vq_space_dim, padding=0),
                                 )

    def decode_clusters(self, latents, N, H, W, C):
        action = -1
        reward = -1
        #z_q_x, x_tilde = self.decode_clusters(latents, N, H, W, C)
        z_q_x = self.embedding(latents.view(latents.shape[0], -1))
        # back to NCHW (orig) - now cluster centers/class
        z_q_x = z_q_x.view(N, H, W, C).permute(0, 3, 1, 2)
        # put quantized data through decoder
        x_tilde = self.decoder(z_q_x)
        # Move prediction to the z_q_x from z_e_x so that I can decode forward
        if self.n_actions > 0:
            action = F.log_softmax(self.action_conv(z_q_x)[:,:,0,0], dim=1)
        # can predict value or reward
        if self.int_reward > 0:
            reward = F.log_softmax(self.int_reward_conv(z_q_x)[:,:,0,0], dim=1)
        if self.reward_value:
            reward = self.reward_val_conv(z_q_x)[:,0,0,0]
        return x_tilde, z_e_x, z_q_x, latents, action, reward
        #return z_q_x, x_tilde

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
        # num_clusters=512, num_z=64
        # measure is of shape bs,10,10,64,512
        # square each element, then sum over channels
        # take sum over each z - find min
        dists = torch.pow(measure, 2).sum(-2)
        # pytorch gives real min and arg min - select argmin
        # this is the closest k for each sample - Equation 1
        # latents is a array of integers
        latents = dists.min(-1)[1]
        # look up cluster centers
        return self.decode_clusters(latents, N, H, W, C)
        #z_q_x, x_tilde = self.decode_clusters(latents, N, H, W, C)
        ## Move prediction to the z_q_x from z_e_x so that I can decode forward
        #if self.n_actions > 0:
        #    action = F.log_softmax(self.action_conv(z_q_x)[:,:,0,0], dim=1)
        ## can predict value or reward
        #if self.int_reward > 0:
        #    reward = F.log_softmax(self.int_reward_conv(z_q_x)[:,:,0,0], dim=1)
        #if self.reward_value:
        #    reward = self.reward_val_conv(z_q_x)[:,0,0,0]
        #return x_tilde, z_e_x, z_q_x, latents, action, reward

if __name__ == '__main__':
    use_cuda = False
    ysize, xsize = 40,40
    if use_cuda:
        model = VQVAE().cuda()
        x = Variable(torch.randn(32,1,ysize,xsize).cuda(), requires_grad=False)
    else:
        model = VQVAE()
        x = Variable(torch.randn(32,1,ysize,xsize), requires_grad=False)

    model.zero_grad()
    x_tilde, z_e_x, z_q_x = model(x)
    z_q_x.retain_grad()

    # losses

    #loss1 = F.binary_cross_entropy(x_tilde, x)
    loss1 = discretized_mix_logistic_loss(x_tilde,2*x-1)
    loss1.backward(retain_graph=True)
    # make sure that encoder is not receiving gradients - only train decoder
    assert model.encoder[-2].bias.grad is None
    model.embedding.zero_grad()
    # straight-thru trick to skip discrete zs
    z_e_x.backward(z_q_x.grad, retain_graph=True)
    # make sure embedding has no gradient
    assert model.embedding.weight.grad.sum().data.cpu().numpy()[0] == 0
    bias = deepcopy(model.encoder[-2].bias.grad.data)

    # detach is like stop gradient
    loss2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss2.backward(retain_graph=True)
    emb = deepcopy(model.embedding.weight.grad.data)
    assert (bias == model.encoder[-2].bias.grad.data).all() is True

    # commitment loss
    Beta = 0.25
    loss3 = Beta*F.mse_loss(z_e_x, z_q_x.detach())
    loss3.backward()
    assert (emb == model.embedding.weight.grad.data).all() is True

    print(loss1, loss2, loss3)


