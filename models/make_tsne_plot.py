#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
#import numpy as np
#
#import mpld3
#from mpld3 import plugins
#
#fig, ax = plt.subplots()
#
#x = np.linspace(-2, 2, 20)
#y = x[:, None]
#X = np.zeros((20, 20, 4))
#
#X[:, :, 0] = np.exp(- (x - 1) ** 2 - (y) ** 2)
#X[:, :, 1] = np.exp(- (x + 0.71) ** 2 - (y - 0.71) ** 2)
#X[:, :, 2] = np.exp(- (x + 0.71) ** 2 - (y + 0.71) ** 2)
#X[:, :, 3] = np.exp(-0.25 * (x ** 2 + y ** 2))
#
#im = ax.imshow(X, extent=(10, 20, 10, 20),
#               origin='lower', zorder=1, interpolation='nearest')
#fig.colorbar(im, ax=ax)
#
#ax.set_title('An Image', size=20)
#
#plugins.connect(fig, plugins.MousePosition(fontsize=14))



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import base64
#import mplcursors
import numpy as np
import os
import sys
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import mpld3
from cv2 import resize

from IPython.display import HTML
from IPython import embed
random_state = np.random.RandomState(4)

import torch
from train_breakout_conv_acn_bce_reconstruction import ConvVAE, make_state
sys.path.append('../agents')
from replay import ReplayMemory

DEVICE = 'cpu'

# TODO - colorgradient related to time in episode
# add step in episode number to replay buffer
# load pretrained model
model_loadpath = '../../model_savedir/results_train_breakout_acn_bce_binary_v2/sigcacn_breakout_binary_0049805312ex.pkl'
# data buffer from experience replay of rl agent
buffer_loadpath = '/usr/local/data/jhansen/planning/model_savedir/MFBreakout_train_anneal_14342_04/breakout_S014342_N0005880131_eval_006000.npz'

# load model info
load_dict = torch.load(model_loadpath, map_location=lambda storage, loc:storage)
# load data
data_buffer = ReplayMemory(load_file=buffer_loadpath)

# args given when model was trained
_args = load_dict['info']['args'][-1]
code_length = _args.code_length
vae_model = ConvVAE(code_length, input_size=1, num_output_channels=1)
vae_model.to(DEVICE)

def tsne_plot(X, images, color, num_clusters=30, perplexity=5, serve_port=8104, html_out_path='mpld3.html'):
    print('computing TSNE')
    Xtsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)
    x = Xtsne[:,0]
    y = Xtsne[:,1]
    # get color from kmeans cluster
    #print('computing KMeans clustering')
    #Xclust = KMeans(n_clusters=num_clusters).fit_predict(Xtsne)
    #c = Xclust


    # Create list of image URIs
    html_imgs = []
    for nidx in range(npimgs.shape[0]):
        f,ax = plt.subplots(figsize=(16,16))
        ax.imshow(resize(npimgs[nidx], (200,200)))
        ax.set_title(batch_idxs[nidx])
        #fname = '%05d.html'%nidx
        dd = mpld3.fig_to_dict(f)
        img = dd['axes'][0]['images'][0]['data']
        html = '<img src="data:image/png;base64,{0}">'.format(img)
        html_imgs.append(html)
        plt.close()

    # Define figure and axes, hold on to object since they're needed by mpld3
    fig, ax = plt.subplots(figsize=(8,8))

    # Make scatterplot and label axes, title
    sc = ax.scatter(x,y,s=100,alpha=0.7, c=color, edgecolors='none')
    plt.title("TSNE")

    # Create the mpld3 HTML tooltip plugin
    tooltip = mpld3.plugins.PointHTMLTooltip(sc, html_imgs)
    # Connect the plugin to the matplotlib figure
    mpld3.plugins.connect(fig, tooltip)
    # Uncomment to save figure to html file
    out=mpld3.fig_to_html(fig)
    fpath = open(html_out_path, 'w')
    fpath.write(out)
    # display is used in jupyter
    #mpld3.display()
    mpld3.show(port=serve_port, open_browser=False)

data_buffer.reset_unique()
batch = data_buffer.get_unique_minibatch(300)
batch_idxs =  batch[-1]
npimgs = batch[3][:,-1]
states, actions, rewards, next_states = make_state(batch[:-1], DEVICE, 255.)

data = next_states[:,-1:]
with torch.no_grad():
    yhat_batch, u_q, s_q = vae_model(data)
    code_book = u_q.cpu().numpy()

tsne_plot(X=code_book, images=npimgs, color=batch_idxs)


