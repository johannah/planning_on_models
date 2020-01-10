import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from glob import glob
from shutil import copyfile

import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from IPython import embed


class IndexedDataset(Dataset):
    def __init__(self, dataset_function, path, train=True, download=True, transform=transforms.ToTensor()):
        """ class to provide indexes into the data -- needed for ACN prior
        """
        self.indexed_dataset = dataset_function(path,
                             download=download,
                             train=train,
                             transform=transform)

    def __getitem__(self, index):
        data, target = self.indexed_dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.indexed_dataset)

def save_checkpoint(state, filename='model.pt'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)

def create_mnist_datasets(dataset_name, base_datadir, batch_size, dataset_transforms):
    dataset = eval('datasets.'+dataset_name)
    datadir = os.path.join(base_datadir, dataset_name)
    train_data = IndexedDataset(dataset, path=datadir,
                                train=True, download=True,
                                transform=dataset_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = IndexedDataset(dataset, path=datadir,
                               train=False, download=True,
                               transform=dataset_transforms)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    data_dict = {'train':train_loader, 'valid':valid_loader}
    nchans,hsize,wsize = data_dict['train'].dataset[0][0].shape
    size_training_set = len(train_data)
    return data_dict, size_training_set, nchans, nchans, hsize, wsize

def create_new_info_dict(arg_dict, base_filepath, base_file):
    if not os.path.exists(base_filepath):
        os.makedirs(base_filepath)
    info = {'base_file':base_file,
            'train_cnts':[],
            'train_losses':{},
            'valid_losses':{},
            'save_times':[],
            'args':[arg_dict],
            'last_save':0,
            'last_plot':0,
            'epoch_cnt':0,
            'base_filepath':base_filepath,
             }
    for arg,val in arg_dict.items():
        info[arg] = val
    if info['cuda']:
        info['device'] = 'cuda'
    else:
        info['device'] = 'cpu'
    return info

def seed_everything(seed=394, max_threads=2):
    torch.manual_seed(394)
    torch.set_num_threads(max_threads)

def plot_example(img_filepath, example, plot_on=[], num_plot=10):
    '''
    img_filepath: location to write .png file
    example: dict with torch images of the same shape [bs,c,h,w] to write
    plot_on: list of keys of images in example dict to write - if blank, plot all keys in example in alphabetical order
    num_plot: limit the number of examples from bs to this int
    '''
    if not len(plot_on):
        # plot all
        plot_on = sorted(example.keys())
    for cnt, pon in enumerate(plot_on):
        bs,c,h,w = example[pon].shape
        num_plot = min([bs, num_plot])
        eimgs = example[pon].view(bs,c,h,w)[:num_plot]
        print('plotting', pon, eimgs.min(), eimgs.max())
        if not cnt:
            comparison = eimgs
        else:
            comparison = torch.cat([comparison, eimgs])
    save_image(comparison.cpu(), img_filepath, nrow=num_plot)
    print('writing comparison image: %s img_path'%img_filepath)

def count_parameters(model):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    num =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def write_log_files(info):
    basename = os.path.split(info['base_filepath'])[1]
    info_filepath = os.path.join(info['base_filepath'],"%s_info.txt"%(basename))
    fp = open(info_filepath, 'w')
    for key, item in info.items():
        if 'loss' or 'cnt' not in key:
            fp.write("%s:%s\n"%(key,item))
    fp.close()
    files = glob(os.path.join(os.path.split(__file__)[0],'*.py'))
    print('making backup of py files')
    bdir = os.path.join(info['base_filepath'], 'py')
    if not os.path.exists(bdir): os.makedirs(bdir)
    for f in files:
        fname = os.path.split(f)[1]
        to_path = os.path.join(bdir, fname)
        copyfile(f, to_path)
        print(f, to_path)

def plot_losses(train_cnts, train_losses, test_losses, name='loss_example.png', rolling_length=4):
    nf = len(train_losses.keys())
    f,ax=plt.subplots(1,nf,figsize=(nf*2,5))
    cmap = matplotlib.cm.get_cmap('viridis')
    color_idxs = np.linspace(.1,.9,num=len(train_losses.keys()))
    colors = np.array([cmap(ci) for ci in color_idxs])
    for idx, key in enumerate(sorted(train_losses.keys())):
        ax[idx].plot(rolling_average(train_cnts, rolling_length),
                rolling_average(train_losses[key], rolling_length),
                lw=1, c=colors[idx])
        ax[idx].plot(rolling_average(train_cnts, rolling_length),
                rolling_average(test_losses[key], rolling_length),
                lw=1, c=colors[idx])
        ax[idx].scatter(rolling_average(train_cnts, rolling_length),
               rolling_average(train_losses[key], rolling_length),
                s=15, c=tuple(colors[idx][None]), marker='x', label='train')
        ax[idx].scatter(rolling_average(train_cnts, rolling_length),
               rolling_average(test_losses[key], rolling_length),
                s=15, c=tuple(colors[idx][None]), marker='o', label='valid')
        ax[idx].set_title(key)
        ax[idx].legend()
    plt.savefig(name)
    plt.close()


def pca_plot(X, images, color, serve_port=8104, html_out_path='mpld3.html', serve=False):
    from sklearn.decomposition import PCA
    import mpld3
    from skimage.transform import resize

    print('computing pca')
    Xpca = PCA(n_components=2).fit_transform(X)
    x = Xpca[:,0]
    y = Xpca[:,1]
    # get color from kmeans cluster
    #print('computing KMeans clustering')
    #Xclust = KMeans(n_clusters=num_clusters).fit_predict(Xtsne)
    #c = Xclust
    # Create list of image URIs
    html_imgs = []
    print('adding hover images')
    for nidx in range(images.shape[0]):
        f,ax = plt.subplots()
        ax.imshow(resize(images[nidx], (180,180)))
        dd = mpld3.fig_to_dict(f)
        img = dd['axes'][0]['images'][0]['data']
        html = '<img src="data:image/png;base64,{0}">'.format(img)
        html_imgs.append(html)
        plt.close()

    # Define figure and axes, hold on to object since they're needed by mpld3
    fig, ax = plt.subplots(figsize=(8,8))
    # Make scatterplot and label axes, title
    sc = ax.scatter(x, y, s=100,alpha=0.7, c=color, edgecolors='none')
    plt.title("PCA")
    # Create the mpld3 HTML tooltip plugin
    tooltip = mpld3.plugins.PointHTMLTooltip(sc, html_imgs)
    # Connect the plugin to the matplotlib figure
    mpld3.plugins.connect(fig, tooltip)
    #plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom())
    # Uncomment to save figure to html file
    out=mpld3.fig_to_html(fig)
    print('writing pca image to %s'%html_out_path)
    fpath = open(html_out_path, 'w')
    fpath.write(out)
    # display is used in jupyter
    #mpld3.display()
    if serve==True:
        mpld3.show(port=serve_port, open_browser=False)



def tsne_plot(X, images, color, perplexity=5, serve_port=8104, html_out_path='mpld3.html', serve=False):
    from sklearn.manifold import TSNE
    import mpld3
    from skimage.transform import resize

    print('computing TSNE')
    Xtsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)
    x = Xtsne[:,0]
    y = Xtsne[:,1]
    # Create list of image URIs
    html_imgs = []
    print('adding hover images')
    for nidx in range(images.shape[0]):
        f,ax = plt.subplots()
        ax.imshow(resize(images[nidx], (180,180)))
        dd = mpld3.fig_to_dict(f)
        img = dd['axes'][0]['images'][0]['data']
        html = '<img src="data:image/png;base64,{0}">'.format(img)
        html_imgs.append(html)
        plt.close()

    # Define figure and axes, hold on to object since they're needed by mpld3
    fig, ax = plt.subplots(figsize=(8,8))
    # Make scatterplot and label axes, title
    sc = ax.scatter(x, y, s=100,alpha=0.7, c=color, edgecolors='none')
    plt.title("TSNE")
    # Create the mpld3 HTML tooltip plugin
    tooltip = mpld3.plugins.PointHTMLTooltip(sc, html_imgs)
    # Connect the plugin to the matplotlib figure
    mpld3.plugins.connect(fig, tooltip)
    #plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom())
    # Uncomment to save figure to html file
    out=mpld3.fig_to_html(fig)
    print('writing tsne image to %s'%html_out_path)
    fpath = open(html_out_path, 'w')
    fpath.write(out)
    # display is used in jupyter
    #mpld3.display()
    if serve==True:
        mpld3.show(port=serve_port, open_browser=False)

##################################################################

def set_model_mode(model_dict, phase):
    for name, model in model_dict.items():
        if name != 'opt':
            #print('setting', name, phase)
            if phase == 'valid':
                model_dict[name].eval()
            else:
                model_dict[name].train()
    return model_dict

def kl_normal(mu_0, log_var_0, mu_1, log_var_1):
    # from https://gist.github.com/Kaixhin/9c90221aae216f59ca2594ea000b0afe
    kl = (2 * (log_var_1 - log_var_0)).exp() + ((mu_1 - mu_0) / log_var_0.exp()) ** 2 - 2 * (log_var_1 - log_var_0) - 1
    return 0.5 * kl.sum(1).mean()

def kl_loss_function(u_q, s_q, u_p, s_p, reduction='sum'):
    ''' reconstruction loss + coding cost
     coding cost is the KL divergence bt posterior and conditional prior
     All inputs are 2d
     Args:
         u_q:  mean of model posterior
         s_q: log std of model posterior
         u_p: mean of conditional prior
         s_p: log std of conditional prior

     Returns: loss
     '''
    acn_KLD = (s_p-s_q-0.5 + ((2*s_q).exp() + (u_q-u_p).pow(2)) / (2*(2*s_p).exp()))
    bs,code_length = u_q.shape
    acn_KLD = acn_KLD.sum(dim=-1)
    if reduction == 'sum':
        return acn_KLD.sum()
    elif reduction == 'mean':
        return acn_KLD.mean()
    else:
        raise ValueError('invalid kl reduction')

def discretized_mix_logistic_loss(prediction, target, nr_mix=10, reduction='mean'):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    Args:
        prediction: model prediction. channels of model prediction should be mean
                    and scale for each channel and weighting bt components --> (2*nr_mix+nr_mix)*num_channels
        target: min/max should be -1 and 1
    **** code for this function from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    """
    chan = prediction.shape[1]
    #assert (prediction.max()<=1 and prediction.min()>=-1)
    assert (target.max()<=1 and target.min()>=-1)
    device = target.device
    # Pytorch ordering
    l = prediction
    x = target
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    #ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    #nr_mix = int(ls[-1] / 10)
    # l is prediction
    logit_probs = l[:, :, :, :nr_mix]

    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix*2]) # 3--changed to 1 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    #x = x.unsqueeze(-1) + torch.Variable(torch.zeros(xs + [nr_mix]).to(device), requires_grad=False)
    x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix], requires_grad=False).to(device)

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    lse = log_sum_exp(log_probs)
    if reduction == 'mean':
        dml_loss = -lse.mean()
    elif reduction == 'sum':
        dml_loss = -lse.sum()
    elif reduction == None:
        dml_loss = -lse
    else:
        raise ValueError('reduction not known')
    return dml_loss

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda : one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot

def sample_from_discretized_mix_logistic(l, nr_mix, only_mean=True, deterministic=False, sampling_temperature=1.0):
    """
    TODO  explain input
    **** code for this function from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    l should be bt -1 and 1

    """
    sampling_temperature = float(sampling_temperature)
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    #from IPython import embed; embed()
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])
    # sample mixture indicator from softmax
    noise = torch.FloatTensor(logit_probs.size())
    if l.is_cuda : noise = noise.cuda()
    noise.uniform_(1e-5, 1. - 1e-5)
    # hack to make deterministic JRH
    # could also just take argmax of logit_probs
    if deterministic or only_mean:
        # make temp small so logit_probs dominates equation
        sampling_temperature = 1e-6
    # sampling temperature from kk
    # https://gist.github.com/kastnerkyle/ea08e1aed59a0896e4f7991ac7cdc147
    # discussion on gumbel sm sampling -
    # https://github.com/Rayhane-mamah/Tacotron-2/issues/155
    noise = (logit_probs.data/sampling_temperature) - torch.log(- torch.log(noise))
    _, argmax = noise.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda : u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    # hack to make deterministic
    if deterministic:
        u= u*0.0+0.5
    if only_mean:
        x = means
    else:
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    out = torch.clamp(torch.clamp(x,min=-1.),max=1.)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow
    **** code for this function from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow
    **** code for this function from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


