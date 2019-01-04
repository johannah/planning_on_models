import os
import config
from glob import glob

model_dirs = sorted(glob(os.path.join(config.model_savedir, '*')))
for d in model_dirs:
    if os.path.isdir(d):
        if not len(glob(os.path.join(d, '*.*'))):
            print('remove', d)
            os.rmdir(d)

