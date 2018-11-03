"""
the test for multi atlas segmentation based on Voxelmorph and Neuron

"""


import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from scipy import stats
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

 # read atlas
 atlas_vol1, atlas_seg1 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/990114_vc722.npz',
                                                              '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/990114_vc722.npz')# [1,160,192,224,1]
 atlas_seg1 = atlas_seg1[0,:,:,:,0]# reduce the dimension to [160,192,224]

 atlas_vol2, atlas_seg2 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/990210_vc792.npz',
                                                              '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/990210_vc792.npz')
 atlas_seg2 = atlas_seg2[0, :, :, :, 0]

 atlas_vol3, atlas_seg3 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/990405_vc922.npz',
                                                              '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/990405_vc922.npz')
 atlas_seg3 = atlas_seg3[0, :, :, :, 0]

 atlas_vol4, atlas_seg4 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/991006_vc1337.npz',
                                                              '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/991006_vc1337.npz')
 atlas_seg4 = atlas_seg4[0, :, :, :, 0]

 atlas_vol5, atlas_seg5 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/991120_vc1456.npz',
                                                              '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/991120_vc1456.npz')
 atlas_seg5 = atlas_seg5[0, :, :, :, 0]
 #atlas = np.load('../data/atlas_norm.npz')
 #atlas_vol = atlas['vol']
 #print('the size of atlas:')
 #print(atlas_vol.shape)
 #atlas_seg = atlas['seg']
 #atlas_vol = np.reshape(atlas_vol, (1,)+atlas_vol.shape+(1,))

 #gpu = '/gpu:' + str(gpu_id)
 os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
 #config = tf.ConfigProto()
 #config.gpu_options.allow_growth = True
 #config.allow_soft_placement = True
 #set_session(tf.Session(config=config))

 # load weights of model
 with tf.device(gpu):
    net = networks.unet(vol_size, nf_enc, nf_dec)
    net.load_weights('/home/ys895/MAS_Models/'+str(iter_num)+'.h5')
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
    pred1 = net.predict([atlas_vol1, X_vol])
    pred2 = net.predict([atlas_vol2, X_vol])
    pred3 = net.predict([atlas_vol3, X_vol])
    pred4 = net.predict([atlas_vol4, X_vol])
    pred5 = net.predict([atlas_vol5, X_vol])
 # Warp segments with flow
 flow1 = pred1[1][0, :, :, :, :]# (1, 160, 192, 224, 3)
 flow2 = pred2[1][0, :, :, :, :]
 flow3 = pred3[1][0, :, :, :, :]
 flow4 = pred4[1][0, :, :, :, :]
 flow5 = pred5[1][0, :, :, :, :]

 sample1 = flow1+grid
 sample1 = np.stack((sample1[:, :, :, 1], sample1[:, :, :, 0], sample1[:, :, :, 2]), 3)
 sample2 = flow2+grid
 sample2 = np.stack((sample2[:, :, :, 1], sample2[:, :, :, 0], sample2[:, :, :, 2]), 3)
 sample3 = flow3+grid
 sample3 = np.stack((sample3[:, :, :, 1], sample3[:, :, :, 0], sample3[:, :, :, 2]), 3)
 sample4 = flow4+grid
 sample4 = np.stack((sample4[:, :, :, 1], sample4[:, :, :, 0], sample4[:, :, :, 2]), 3)
 sample5 = flow5+grid
 sample5 = np.stack((sample5[:, :, :, 1], sample5[:, :, :, 0], sample5[:, :, :, 2]), 3)

 warp_seg1 = interpn((yy, xx, zz), atlas_seg1[ :, :, : ], sample1, method='nearest', bounds_error=False, fill_value=0) # (160, 192, 224)
 warp_seg2 = interpn((yy, xx, zz), atlas_seg2[:, :, :], sample2, method='nearest', bounds_error=False, fill_value=0)
 warp_seg3 = interpn((yy, xx, zz), atlas_seg3[:, :, :], sample3, method='nearest', bounds_error=False, fill_value=0)
 warp_seg4 = interpn((yy, xx, zz), atlas_seg4[:, :, :], sample4, method='nearest', bounds_error=False, fill_value=0)
 warp_seg5 = interpn((yy, xx, zz), atlas_seg5[:, :, :], sample5, method='nearest', bounds_error=False, fill_value=0)


 # label fusion: get the final warp_seg
 warp_seg = np.empty((160, 192, 224))
 for x in range(0,160):
     for y in range(0,192):
         for z in range(0,224):
             warp_arr = np.array([[warp_seg1[x,y,z]],[warp_seg2[x,y,z]],[warp_seg3[x,y,z]],[warp_seg4[x,y,z]],[warp_seg5[x,y,z]]])
             warp_seg[x,y,z] = stats.mode(warp_arr)




 vals, _ = dice(warp_seg, X_seg[0,:,:,:,0], labels=labels, nargout=2)
 print(np.mean(vals), np.std(vals))


 # plot the outcome of warp seg
 #warp_seg = warp_seg.reshape((warp_seg.shape[1], warp_seg.shape[2], warp_seg.shape[0]))
 #warp_seg2 = np.empty(shape = (warp_seg.shape[1], warp_seg.shape[2], warp_seg.shape[0]))
 #for i in range(0,warp_seg.shape[1]):
 # warp_seg2[i,:,:] = np.transpose(warp_seg[:,i,:])
 #nu.plot.slices(warp_seg)


if __name__ == "__main__":
	test(sys.argv[1], sys.argv[2])