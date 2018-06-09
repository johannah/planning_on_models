import torch
import numpy as np
from IPython import embed
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os, sys
from imageio import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import img_as_ubyte
oy = 15
# height - oyb
ox = 9
chicken_color = 240
stripes = 214
road = 142
staging = 170
input_ysize = input_xsize = 80
orig_ysize = 210
orig_xsize = 160
# i dont think this changes 
extra_chicken = np.array([[77, 78, 78, 78, 79], [54, 52, 53, 54, 54]]) 
import matplotlib.pyplot as plt
def prepare_img(obs):
    # turn to gray
    cropped = obs[oy:(orig_ysize-oy),ox:]
    gimg = img_as_ubyte(rgb2gray(cropped))
    gimg[stripes == gimg] = 0
    gimg[road == gimg] = 0
    gimg[staging == gimg] = 0
    print(np.where(gimg == chicken_color))
    sgimg = img_as_ubyte(resize(gimg, (input_ysize, input_xsize), order=0))
    sgimg[extra_chicken[0], extra_chicken[1]] = 0
    our_chicken = np.where(sgimg == chicken_color)
    sgimg[our_chicken[0], our_chicken[1]] = 0
    return sgimg, our_chicken

def undo_img_scaling(sgimg, our_chicken):
    sgimg[our_chicken] = chicken_color
    rec = np.zeros((orig_ysize, orig_xsize))
    outimg = img_as_ubyte(resize(sgimg, (orig_ysize-(oy*2), orig_xsize-ox), order=0))
    rec[oy:(orig_ysize-oy),ox:] = outimg
    return rec

def transform_freeway(obs):
    h,w,c = obs.shape
    o = obs[25:(h-26),9:,:]
    cr = img_as_ubyte(rgb2gray(resize(o, (80,80), order=0)))
    return cr

def remove_background(gimg):
    # background has two colors - 142 is the asphalt and 214 is the lines
    gimg[gimg==road] = 0
    gimg[gimg==stripes] = 0
    return gimg

def remove_chicken(gimg):
    # remove chicken from background removed image
    chicken = np.where(gimg==chicken_color)
    gimg[chicken] = 0
    return gimg, chicken


class FroggerDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None, max_pixel_used=254.0, min_pixel_used=0.0):
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, '*.png')
        self.max_pixel_used = max_pixel_used
        self.min_pixel_used = min_pixel_used
        ss = sorted(glob(search_path))
        self.indexes = [s for s in ss if 'gen' not in s]
        print("found %s files in %s" %(len(self.indexes), search_path))

        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            sys.exit()
        if limit > 0:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]
            print('limited to first %s examples' %len(self.indexes))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_name = self.indexes[idx]
        image = imread(img_name)
        if len(image.shape) == 2:
            image = image[:,:,None].astype(np.float32)
        if self.transform is not None:
            # bt 0 and 1
            image = (self.transform(image)-self.min_pixel_used)/float(self.max_pixel_used-self.min_pixel_used)

        return image,img_name

#class FlattenedFroggerDataset(Dataset):
#    def __init__(self, root_dir, transform=None, limit=None):
#        self.root_dir = root_dir
#        self.transform = transform
#        search_path = os.path.join(self.root_dir, 'seed_*.png')
#        ss = sorted(glob(search_path))
#        self.indexes = [s for s in ss if 'gen' not in s]
#        print("found %s files in %s" %(len(self.indexes), search_path))
#
#        if not len(self.indexes):
#            print("Error no files found at {}".format(search_path))
#            raise
#        if limit > 0:
#            self.indexes = self.indexes[:min(len(self.indexes), limit)]
#            print('limited to first %s examples' %len(self.indexes))
#
#    def __len__(self):
#        return len(self.indexes)
#
#    def __getitem__(self, idx):
#        img_name = self.indexes[idx]
#        image = imread(img_name)
#        image = image[:,:,None].astype(np.float32)
#        # TODO - this is non-normalized
#        image = np.array(image.ravel())
#        return image,img_name
#
#
#z_q_x_mean = 0.16138
#z_q_x_std = 0.7934
#
#class VqvaeDataset(Dataset):
#    def __init__(self, root_dir, transform=None, limit=None):
#        self.root_dir = root_dir
#        self.transform = transform
#        search_path = os.path.join(self.root_dir, '*z_q_x.npy')
#        self.indexes = sorted(glob(search_path))
#        print("found %s files in %s" %(len(self.indexes), search_path))
#        if not len(self.indexes):
#            print("Error no files found at {}".format(search_path))
#            raise
#        if limit > 0:
#            self.indexes = self.indexes[:min(len(self.indexes), limit)]
#            print('limited to first %s examples' %len(self.indexes))
#
#    def __len__(self):
#        return len(self.indexes)
#
#    def __getitem__(self, idx):
#        data_name = self.indexes[idx]
#        data = np.load(data_name).ravel().astype(np.float32)
#        # normalize v_q
#        data = (data-z_q_x_mean)/z_q_x_std
#        # normalize for embedding space
#        #data = 2*((data/512.0)-0.5)
#        return data,data_name
#
#class EpisodicVaeFroggerDataset(Dataset):
#    def __init__(self, root_dir, transform=None, limit=-1, search='*conv_vae.npz'):
#        # what really matters is the seed - only generated one game per seed
#        #seed_00334_episode_00029_frame_00162.png
#        self.root_dir = root_dir
#        self.transform = transform
#        search_path = os.path.join(self.root_dir, search)
#        self.indexes = sorted(glob(search_path))
#        print("will use transform:%s"%transform)
#        print("found %s files in %s" %(len(self.indexes), search_path))
#        if not len(self.indexes):
#            print("Error no files found at {}".format(search_path))
#            sys.exit()
#        if limit > 0:
#            self.indexes = self.indexes[:min(len(self.indexes), limit)]
#            print('limited to first %s examples' %len(self.indexes))
#
#    def __len__(self):
#        return len(self.indexes)
#
#    def __getitem__(self, idx):
#        dname = self.indexes[idx]
#        d = np.load(open(dname, 'rb'))
#        mu = d['mu'].astype(np.float32)[:,best_inds]
#        sig = d['sigma'].astype(np.float32)[:,best_inds]
#        if self.transform == 'pca':
#            if not idx:
#                print("tranforming dataset using pca")
#            mu_scaled = mu-vae_mu_mean
#            mu_scaled = (np.dot(mu_scaled, V.T)/Xpca_std).astype(np.float32)
#        elif self.transform == 'std':
#            mu_scaled= ((mu-vae_mu_mean)/vae_mu_std).astype(np.float32)
#        else:
#            mu_scaled = mu
#            sig_scaled = sig
#
#        return mu_scaled,mu,sig_scaled,sig,dname
#
#
#class EpisodicDiffFroggerDataset(Dataset):
#    def __init__(self, root_dir, transform=None, limit=-1, search='*conv_vae.npz'):
#        # what really matters is the seed - only generated one game per seed
#        #seed_00334_episode_00029_frame_00162.png
#        self.root_dir = root_dir
#        self.transform = transform
#        search_path = os.path.join(self.root_dir, search)
#        self.indexes = sorted(glob(search_path))
#        dparams = np.load('vae_diff_params.npz')
#        self.mu_diff_mean = dparams['mu_diff_mean'][best_inds]
#        self.mu_diff_std = dparams['mu_diff_std'][best_inds]
#        self.sig_diff_mean = dparams['sig_diff_mean'][best_inds]
#        self.sig_diff_std = dparams['sig_diff_std'][best_inds]
#        print("will use transform:%s"%transform)
#        print("found %s files in %s" %(len(self.indexes), search_path))
#        if not len(self.indexes):
#            print("Error no files found at {}".format(search_path))
#            sys.exit()
#        if limit > 0:
#            self.indexes = self.indexes[:min(len(self.indexes), limit)]
#            print('limited to first %s examples' %len(self.indexes))
#
#    def __len__(self):
#        return len(self.indexes)
#
#    def __getitem__(self, idx):
#        if idx == 0:
#            print("loading first file")
#        dname = self.indexes[idx]
#        d = np.load(open(dname, 'rb'))
#        mu = d['mu'].astype(np.float32)[:,best_inds]
#        sig = d['sigma'].astype(np.float32)[:,best_inds]
#        mu_diff = np.diff(mu,n=1,axis=0)
#        sig_diff = np.diff(sig,n=1,axis=0)
#        if self.transform == 'std':
#            if not idx:
#                print("performing transform std")
#            mu_diff_scaled= ((mu_diff-self.mu_diff_mean)/self.mu_diff_std).astype(np.float32)
#            sig_diff_scaled= ((sig_diff-self.sig_diff_mean)/self.sig_diff_std).astype(np.float32)
#            # how to unscale
#            #mu_diff_unscaled = ((mu_scaled*self.mu_diff_std)+self.mu_diff_mean).astype(np.float32)
#            #sig_diff_unscaled = ((sig_scaled*self.sig_diff_std)+self.sig_diff_mean).astype(np.float32)
#        else:
#            if not idx:
#                print("performing no data transform")
#            mu_diff_scaled = mu_diff
#            sig_diff_scaled = sig_diff
#        return mu_diff_scaled,mu_diff,mu,sig_diff_scaled,sig_diff,sig,dname
#
#
class EpisodicVqVaeFroggerDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=-1, search='seed*episode*.npz'):
        # what really matters is the seed - only generated one game per seed
        #seed_00334_episode_00029_frame_00162.png
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, search)
        self.indexes = sorted(glob(search_path))
        print("will use transform:%s"%transform)
        print("found %s files in %s" %(len(self.indexes), search_path))
        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            sys.exit()
        if limit > 0:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]
            print('limited to first %s examples' %len(self.indexes))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        dname = self.indexes[idx]
        d = np.load(open(dname, 'rb'))
        latents = d['latents']
        return latents,dname





