import os
import sys

base_datadir = '../../dataset/'
model_savedir = '../../model_savedir'
results_savedir = '../../results'
freeway_train_frames_dir = os.path.join(base_datadir, 'freeway_train_frames')
freeway_test_frames_dir = os.path.join(base_datadir, 'freeway_test_frames/')
freeway_max_pixel = 254.0
freeway_min_pixel = 0.0
freeway_gt_dir = os.path.join(base_datadir, 'freeway_gt')
