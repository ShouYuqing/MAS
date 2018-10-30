"""
the test for single atlas segmentation based on Voxelmorph and Neuron

"""


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


def test(iter_num, gpu_id, vol_size=(160,192,224), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,32,16,16,3]):
 gpu = '/gpu:' + str(gpu_id)

    # Anatomical labels we want to evaluate
 labels = sio.loadmat('../data/labels.mat')['labels'][0]

 atlas = np.load('../data/atlas_norm.npz')
 atlas_vol = atlas['vol']
 print('the size of atlas:')
 print(atlas_vol.shape)
 atlas_seg = atlas['seg']
 atlas_vol = np.reshape(atlas_vol, (1,)+atlas_vol.shape+(1,))

 config = tf.ConfigProto()
 config.gpu_options.allow_growth = True
 config.allow_soft_placement = True
 set_session(tf.Session(config=config))

 # load weights of model
 with tf.device(gpu):
    net = networks.unet(vol_size, nf_enc, nf_dec)
    net.load_weights('/home/ys895/SAS_Models/'+str(iter_num)+'.h5')
    #net.load_weights('../models/' + model_name + '/' + str(iter_num) + '.h5')

 xx = np.arange(vol_size[1])
 yy = np.arange(vol_size[0])
 zz = np.arange(vol_size[2])
 grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4) # (160, 192, 224, 3)
 X_vol, X_seg = datagenerators.load_example_by_name('../data/test_vol.npz', '../data/test_seg.npz')

 # change the direction of the atlas data and volume data
 # pred[0].shape (1, 160, 192, 224, 1)
 # pred[1].shape (1, 160, 192, 224, 3)
 with tf.device(gpu):
    pred = net.predict([atlas_vol, X_vol])
	# Warp segments with flow
 flow = pred[1][0, :, :, :, :] # (1, 160, 192, 224, 3)
 sample = flow+grid
 sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
 warp_seg = interpn((yy, xx, zz), atlas_seg[ :, :, : ], sample, method='nearest', bounds_error=False, fill_value=0) # (160, 192, 224)
 vals, _ = dice(warp_seg, X_seg[0,:,:,:,0], labels=labels, nargout=2)
 print(np.mean(vals), np.std(vals))


# plot the outcome of warp seg
 warp_seg = warp_seg.reshape((warp_seg.shape[2], warp_seg.shape[1], warp_seg.shape[0]))
 nu.plot.slices(warp_seg)


if __name__ == "__main__":
	test(sys.argv[1], sys.argv[2])
