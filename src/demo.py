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
import matplotlib.pyplot as plt

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

X_seg_slice = X_seg[0, :, :, :, 0 ]
#print(X_seg_slice)
#X_seg_slice.reshape([X_seg_slice.shape[0],X_seg_slice.shape[2]])
#print(X_seg_slice.shape)
#X_seg_slice = X_seg_slice.reshape((X_seg_slice.shape[1],X_seg_slice.shape[0],X_seg_slice.shape[2]))
#X_seg_slice = X_seg_slice.reshape((X_seg_slice.shape[2],X_seg_slice.shape[1],X_seg_slice.shape[0]))
#for i in range(0,X_seg_slice.shape[0]):
#    list.insert(X_seg_slice[i,:,:])
#X_seg_slice = X_seg_slice.reshape([X_seg_slice.shape[0],X_seg_slice.shape[1],X_seg_slice.shape[2]])
#X_seg_slice = [X_seg_slice]
#fig,axs = nu.plot.slices(X_seg_slice)
#fig.set_size_inches(width, rows/cols*width)
#plt.tight_layout()
#print(fig.shape)
#fig.savefig("1.pdf")
warp_seg = X_seg_slice
warp_seg = X_seg_slice.reshape((warp_seg.shape[0], warp_seg.shape[1], warp_seg.shape[2]))
#warp_seg2 = np.empty(shape=(warp_seg.shape[1], warp_seg.shape[2], warp_seg.shape[0]))
#for i in range(0, warp_seg.shape[1]):
#    warp_seg2[i, :, :] = np.transpose(warp_seg[:, i, :])
nu.plot.slices(warp_seg)
