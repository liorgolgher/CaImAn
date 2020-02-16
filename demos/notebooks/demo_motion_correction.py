## 3D Motion Correction demo
## modified from https://github.com/flatironinstitute/CaImAn/blob/master/demos/notebooks/demo_motion_correction.ipynb




from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div


import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import scipy
from skimage.external.tifffile import TiffFile
import sys
import time
import logging
#import zarr

try:
    cv2.setNumThreads(0)
except:
    pass
'''
try:
    if __IPYTHON__:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)
'''
import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo

fld = '/data/Lior/lst/2020/2020_01_13/'
#fnames = fld + 'txyz_30vps_4D_dataset.zarr'
fnames = fld + 'txyz_30vps_4D_dataset.HDF5'

print(fnames)

#m_orig = cm.load_movie_chain(fnames)
m_orig = cm.load(fnames) # single movie file
downsample_ratio = .4  # motion can be perceived better when downsampling in time
# The following line is not ready yet for 4D datasets (LG)
# m_orig.resize(downsample_ratio, 1, 1, 1).play(q_max=99.5, fr=30, magnification=2)   # play movie (press q to exit)

max_shifts = (6, 12, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48, 48)  # create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24, 24)  # overlap between pathes (size of patch strides+overlaps)
num_frames_split = 100  # length in frames of each chunk of the movie (to be processed in parallel)
max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
is3D = True # added by LG to support 3D moco

#%% start the cluster (if a cluster already exists terminate it)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)


# create a motion correction object

mc = MotionCorrect(fnames, dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid,
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan, is3D=is3D)


##capture
# correct for rigid motion correction and save the file (in memory mapped form)
mc.motion_correct(save_movie=True)
