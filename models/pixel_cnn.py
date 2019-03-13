# learning pixelcnn -
# largely based on code from Ritesh Kumar - https://github.com/ritheshkumar95/pytorch-vqvae/
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from IPython import embed

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init_xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            pass
            #print('not initializing {}'.format(classname))

class GatedActivation(nn.Module):
    def __init__(self):
        super(GatedActivation, self).__init__()

    def forward(self, x):
        x,y = x.chunk(2,dim=1)
        return torch.tanh(x)*torch.sigmoid(y)

class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=None,
                       spatial_condition_size=None, float_condition_size=None, hsize=28, wsize=28):
        super(GatedMaskedConv2d, self).__init__()
        # ("Kernel size must be odd")
        assert (kernel % 2 == 1 )
        self.mask_type = mask_type
        self.residual = residual
        # unique for every layer of the pixelcnn - takes integer from 0-x and
        # returns slice which is 0 to 1ish - treat each latent value like a
        # "word" embedding
        self.dim = dim
        self.hsize = hsize
        self.wsize = wsize

        #self.class_embedding_size = 2*self.dim*self.hsize*self.wsize
        #self.class_cond_embedding = nn.Embedding(n_classes, self.class_embedding_size)
        vkernel_shape = (kernel//2 + 1, kernel)
        vpadding_shape = (kernel//2, kernel//2)

        cond_kernel_shape = (kernel, kernel)
        cond_padding_shape = (kernel//2, kernel//2)

        hkernel_shape = (1,kernel//2+1)
        hpadding_shape = (0,kernel//2)

        if n_classes is not None:
            self.class_cond_embedding = nn.Embedding(n_classes, 2*dim)
        if float_condition_size is not None:
            self.float_condition_layer = nn.Linear(float_condition_size, 2*self.dim)
        if spatial_condition_size is not None:
            self.spatial_condition_stack = nn.Conv2d(spatial_condition_size, dim*2, kernel_size=cond_kernel_shape, stride=1, padding=cond_padding_shape)

        self.vert_stack = nn.Conv2d(dim, dim*2, kernel_size=vkernel_shape, stride=1, padding=vpadding_shape)
        self.vert_to_horiz = nn.Conv2d(2*dim, 2*dim, kernel_size=1)
        self.horiz_stack = nn.Conv2d(dim, dim*2, kernel_size=hkernel_shape, stride=1, padding=hpadding_shape)
        # kernel_size 1 are "fixup layers" to make things match up
        self.horiz_resid = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:,:,-1].zero_() # mask final row
        self.horiz_stack.weight.data[:,:,:,-1].zero_() # mask final column

    def forward(self, x_v, x_h, class_condition=None, spatial_condition=None, float_condition=None):
        # class condition coming in is just an integer
        # spatial_condition should be the same size as the input
        if self.mask_type == 'A':
            # make first layer causal to prevent cheating
            self.make_causal()
        # manipulation to get same size out of h_vert
        # output of h_vert is 6,6,(2*dim)
        # x_v.shape 32,1,84,84
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:,:,:x_v.shape[-1], :]
        # h_vert is (batch_size,512,6,6)

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:,:,:,:x_h.shape[-2]]
        v2h = self.vert_to_horiz(h_vert)

        input_to_out_v = h_vert
        input_to_out_h = v2h + h_horiz

        if float_condition is not None:
            self.float_condition = torch.autograd.Variable(float_condition, requires_grad=True)
            float_out = self.float_condition_layer(self.float_condition)
            input_to_out_v += float_out[:,:,None,None]
            input_to_out_h += float_out[:,:,None,None]

        # add class conditioning
        if class_condition is not None:
            # no gradient because class_condition is an integer
            self.class_condition = class_condition
            class_condition_emb = self.class_cond_embedding(self.class_condition)
            input_to_out_v += class_condition_emb[:,:,None,None]
            input_to_out_h += class_condition_emb[:,:,None,None]

        if spatial_condition is not None:
            self.spatial_condition = torch.autograd.Variable(spatial_condition, requires_grad=True)
            spatial_c_e = self.spatial_condition_stack(self.spatial_condition)
            input_to_out_v += spatial_c_e
            input_to_out_h += spatial_c_e

        out_v = self.gate(input_to_out_v)
        gate_h = self.gate(input_to_out_h)

        if self.residual:
            out_h = self.horiz_resid(gate_h)+x_h
        else:
            out_h = self.horiz_resid(gate_h)
        return out_v, out_h

class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=512, dim=256, n_layers=15, n_classes=None,
                 spatial_condition_size=None, float_condition_size=None,
                 last_layer_bias=0.0, hsize=28, wsize=28):
        super(GatedPixelCNN, self).__init__()
        self.hsize = hsize
        self.wsize = wsize
        self.dim = dim
        # lookup table to store input
        #self.embedding = nn.Embedding(input_dim, self.dim)
        #if spatial_cond_size is not None:
        #    # assume same vocab size - but input_dim could be different here
        #    self.spatial_cond_embedding = nn.Embedding(input_dim, self.dim)
        # build pixelcnn layers - functions like normal python list, but modules are registered
        self.layers = nn.ModuleList()
        # first block has Mask-A convolution - (no residual connections)
        # subsequent blocks have Mask-B convolutions
        self.layers.append(GatedMaskedConv2d(mask_type='A', dim=self.dim,
                           kernel=7, residual=False, n_classes=n_classes,
                                             spatial_condition_size=spatial_condition_size,
                                             float_condition_size=float_condition_size,
                                             hsize=self.hsize, wsize=self.wsize))
        for i in range(1,n_layers):
            self.layers.append(GatedMaskedConv2d(mask_type='B', dim=self.dim,
                                                 kernel=3, residual=True,
                                                 n_classes=n_classes,
                                                 spatial_condition_size=spatial_condition_size,
                                                 float_condition_size=float_condition_size,
                                                hsize=self.hsize, wsize=self.wsize
                                                 ))

        self.init_conv = nn.Conv2d(input_dim, self.dim, 1)
        self.output_conv = nn.Sequential(
                                         nn.Conv2d(self.dim, 512, 1),
                                         nn.ReLU(True),
                                         nn.Conv2d(512, input_dim, 1)
                                         )

        # in pytorch - apply(fn)  recursively applies fn to every submodule as returned by .children
        # apply xavier_uniform init to all weights
        self.apply(weights_init)
        self.output_conv[-1].bias.data.fill_(last_layer_bias)

    def forward(self, x, class_condition=None, spatial_condition=None, float_condition=None):
        # need self.dim channels - do initial conv
        self.spatial_condition = spatial_condition
        self.float_condition = float_condition
        self.class_condition = class_condition

        xo = self.init_conv(x)
        x_v, x_h = (xo,xo)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v=x_v, x_h=x_h, class_condition=self.class_condition,
                             spatial_condition=self.spatial_condition, float_condition=self.float_condition)
        return self.output_conv(x_h)

