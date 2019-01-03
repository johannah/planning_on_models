import os, sys
import time
import numpy as np
import config
from IPython import embed
from glob import glob
import shutil
if __name__ == '__main__':
    run_num = 0
    glob('*.pkl')

    # move files to dirs
    #undiredfiles = glob(os.path.join(config.model_savedir, '*.*'))
    #for f in undiredfiles:
    #    bname = os.path.split(f)[1].split('_')[0]
    #    newdir = os.path.join(config.model_savedir, bname)
    #    if not os.path.isdir(newdir):
    #        os.makedirs(newdir)
    #    pklfiles = glob(os.path.join(config.model_savedir, '%s*.pkl'%bname))
    #    pngfiles = glob(os.path.join(config.model_savedir, '%s*.png'%bname))
    #    for p in pklfiles+pngfiles:
    #        bp = os.path.split(p)[1]
    #        newp = os.path.join(newdir, bp)
    #        print('moving to new path', newp)
    #        shutil.move(p,newp)

    #exdirs = glob(os.path.join(config.model_savedir, '*ex'))
    #for ex in exdirs:
    #    realdir =  os.path.join(config.model_savedir, ex.split('_')[0])
    #    shutil.move(ex,realdir)



    #model_base_filedir = os.path.join(config.model_savedir, args.savename + '%02d'%run_num)
    #while os.path.exists(model_base_filedir):
    #    run_num +=1
    #    model_base_filedir = os.path.join(config.model_savedir, args.savename + '%02d'%run_num)
    #os.makedirs(model_base_filedir)
    #model_base_filepath = os.path.join(model_base_filedir, args.savename)

