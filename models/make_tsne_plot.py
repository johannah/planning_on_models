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
import mpld3
from skimage.transform import resize

from IPython import embed

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
    sc = ax.scatter(x,y,s=100,alpha=0.7, c=color, edgecolors='none')
    plt.title("TSNE")

    # Create the mpld3 HTML tooltip plugin
    tooltip = mpld3.plugins.PointHTMLTooltip(sc, html_imgs)
    # Connect the plugin to the matplotlib figure
    mpld3.plugins.connect(fig, tooltip)
    #plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom())
    # Uncomment to save figure to html file
    out=mpld3.fig_to_html(fig)
    print('writing to %s'%html_out_path)
    fpath = open(html_out_path, 'w')
    fpath.write(out)
    # display is used in jupyter
    #mpld3.display()
    mpld3.show(port=serve_port, open_browser=False)

