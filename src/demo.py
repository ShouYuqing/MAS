'''
demo for testing the function in the neuron & voxelmorph
'''
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('../ext/medipy-lib')
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')

import medipy
import networks
from medipy.metrics import dice
import datagenerators
import neuron as nu

X_vol, X_seg = datagenerators.load_example_by_name('../data/test_vol.npz', '../data/test_seg.npz') # (160, 192, 224)

X_seg_slice = X_seg[0, :, :, :, 0]
X_seg_slice = X_seg_slice.reshape([X_seg_slice.shape[1],X_seg_slice.shape[0],X_seg_slice.shape[2]])
fig,axs = nu.plot.slices(X_seg_slice)
savefig("1.jpg")
