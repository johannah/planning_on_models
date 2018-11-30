import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vqvae_pcnn_future_model import VQPCNN_model
import numpy as np
import pickle
import os, sys
from IPython import embed
from scipy.misc import imsave
from models import config
from glob import glob
import shutil
obs_dirname = 'obs'
pobs_dirname = 'pobs'
pred_dirname = 'pred'
title_font = {'fontname':'Arial', 'size':'8', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space


def get_data_paths(basepath):
    data_path = basepath + '_data.npz'
    summary_path = basepath + '_summary.pkl'
    return data_path, summary_path

def update_npz_file(basepath, new_dict):
    data_path, summary_path = get_data_paths(basepath)
    np.savez(data_path, **new_dict)

def get_basepath_data(basepath, rerun=False):
    data_path, summary_path = get_data_paths(basepath)
    records = pickle.load(open(summary_path, 'r'))
    arr = np.load(data_path)
    img_path = basepath+'_imgs'
    for i in [os.path.join(img_path, obs_dirname),
              os.path.join(img_path, pred_dirname),
              os.path.join(img_path, pobs_dirname)]:
        if rerun:
            if os.path.exists(i):
                print("removing directory: %s" %i)
                shutil.rmtree(i)
        if not os.path.exists(i):
            os.makedirs(i)

    future_model = ''
    #if records['args'].future_model == 'vqvae_pcnn':
    #    # TODO load from model
    #    future_model = VQPCNN_model(records['DEVICE'],
    #                                config.model_savedir,
    #                                load_vq_name=  records['args'].vq_model_name,
    #                                load_pcnn_name=records['args'].pcnn_model_name,
    #                                dsize=80, nr_logistic_mix=10,
    #                                num_z=64, num_clusters=512,
    #                                N_LAYERS = 10, # layers in pixelcnn
    #                                DIM = 256,
    #                                history_size=records['args'].history_size,
    #                                )
    return records, arr, img_path, future_model

def plot_true_state_actions(records, arr, img_path):
    obs_count = records['obs_count']
    img_names = glob(os.path.join(img_path, obs_dirname, 'O*.png'))
    pobs_names = glob(os.path.join(img_path, pobs_dirname, 'P*.png'))
    for c in obs_count:
        img_name = os.path.join(img_path, obs_dirname, 'O%05d.png' %c)
        pobs_name = os.path.join(img_path, pobs_dirname, 'PO%05d.png' %c)
        if img_name not in img_names:
            if not c%100:
                print("saving %s" %img_name)
            img = arr['observations'][c]
            imsave(img_name, img)
        if pobs_name not in pobs_names:
            if not c%100:
                print("saving %s" %pobs_name)
            pobs = arr['pobservations'][c]
            pobs[records['agent_position_y_min'][c], records['agent_position_x_min'][c]] = 255
            pobs[records['agent_position_y_max'][c], records['agent_position_x_min'][c]] = 255
            pobs[records['agent_position_y_min'][c], records['agent_position_x_max'][c]] = 255
            pobs[records['agent_position_y_max'][c], records['agent_position_x_max'][c]] = 255
            imsave(pobs_name, pobs)

def plot_predictions(basepath, records, arr, img_path, future_model, override_predictions):
    pred_count = records['future_prediction_indexes']
    lpred = arr['lpredictions']
    pobs = arr['pobservations']
    _,obs_xmax,obs_ymax=pobs.shape
    num_preds = lpred.shape[1]
    print(arr.keys())
    if (('ppredictions' not in arr.keys()) or (override_predictions)):
        ppred = np.zeros((len(pred_count), num_preds, pobs.shape[-2], pobs.shape[-1]))
        for cc, ic in enumerate(pred_count):
            print("decoding %s latents from %s prediction %s/%s"%(num_preds, ic, cc+1, len(pred_count)))
            ppred[cc] = future_model.decode_latents(lpred[cc])
        new_dict = dict(arr)
        new_dict['ppredictions'] = ppred
        update_npz_file(basepath, new_dict)
    else:
        ppred = arr['ppredictions']

    for cc, ic in enumerate(pred_count):
        pred_img_path = os.path.join(img_path, pred_dirname, 'P%05d.png' %ic)
        #if not os.path.exists(pred_img_path):
        if 1:
            f, ax = plt.subplots(2, num_preds+1, figsize=(num_preds+1,2))
            #plt.subplots_adjust(wspace=.01,hspace=-.57)
            plt.setp([a.get_xticklabels() for a in ax.ravel()], visible=False)
            plt.setp([a.get_yticklabels() for a in ax.ravel()], visible=False)
            [a.set_xticks([]) for a in ax.ravel()]
            [a.set_yticks([]) for a in ax.ravel()]
            [a.set_xlim([0,obs_xmax-1]) for a in ax.ravel()]
            [a.set_ylim([0,obs_ymax-1]) for a in ax.ravel()]

            ax[0,0].imshow(pobs[ic], origin='lower')
            ax[0,0].set_title('t=%d'%ic, **title_font)
            ax[0,0].set_ylabel('obs', **title_font)

            ax[1,0].imshow(pobs[ic], origin='lower')
            ax[1,0].set_ylabel('pred', **title_font)

            for im in range(1,num_preds+1):
                ax[0,im].imshow(pobs[ic+im], origin='lower')
                ax[0,im].set_title('t+%02d'%im, **title_font)
                ax[1,im].imshow(ppred[cc,im-1], origin='lower')
            plt.tight_layout()
            plt.savefig(pred_img_path)
            embed()
            plt.close()
    return ppred



def newest(search_path):
    files = glob(search_path)
    newest_file = max(files, key=os.path.getctime)
    newest_time = os.path.getctime(newest_file)
    print('Using file: %s created at %s' %(newest_file, newest_time))
    return newest_file

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_name', default='None', type=str, help='''filename of base
                        project to load ie d2018-11-29-T15-53-18_SI000''')
    parser.add_argument('-r', '--base_dir', default=config.results_savedir, type=str, help='''base dir to
                        load project from. Default is %s''' %config.results_savedir)
    parser.add_argument('--run_all', default=False, type=bool, help='''flag to run all project files from the basedir''')
    parser.add_argument('--rerun', default=False, action='store_true', help='''flag to rewrite images even if they are there''')
    parser.add_argument('--override_predictions', default=False, action='store_true', help='''predict new future obs even if they already exist''')

    args = parser.parse_args()

    if args.run_all:
        # search for most recent file in dir
        npz_list = glob(args.base_dir+'*.npz')
        for n in npz_list:
            project_path = n.replace('_data.npz', '')
            records, arr, img_path, future_model = get_basepath_data(project_path, args.rerun)
            plot_true_state_actions(records, arr, img_path)

    else:
        if args.project_name == 'None':
            # search in base dir for most recent
            # get newest path
            newest_path = newest(os.path.join(args.base_dir,'*.npz'))
            project_path = newest_path.replace('_data.npz', '')
        else:
            project_path = os.path.join(args.base_dir, args.project_name)
        records, arr, img_path, future_model = get_basepath_data(project_path, args.rerun)
        plot_true_state_actions(records, arr, img_path)
        plot_predictions(project_path, records, arr, img_path, future_model, override_predictions=args.override_predictions)

