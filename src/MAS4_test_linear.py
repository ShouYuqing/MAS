"""
the test for multi atlas segmentation based on Voxelmorph and Neuron

"""


import os
import sys
import glob

# third party
import tensorflow as tf
import keras
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
 atlas_seg1 = keras.utils.to_categorical(atlas_seg1)

 atlas_vol2, atlas_seg2 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/990210_vc792.npz',
                                                              '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/990210_vc792.npz')
 atlas_seg2 = atlas_seg2[0, :, :, :, 0]
 atlas_seg2 = keras.utils.to_categorical(atlas_seg2)

 atlas_vol3, atlas_seg3 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/990405_vc922.npz',
                                                              '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/990405_vc922.npz')
 atlas_seg3 = atlas_seg3[0, :, :, :, 0]
 atlas_seg3 = keras.utils.to_categorical(atlas_seg3)

 atlas_vol4, atlas_seg4 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/991006_vc1337.npz',
                                                              '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/991006_vc1337.npz')
 atlas_seg4 = atlas_seg4[0, :, :, :, 0]
 atlas_seg4 = keras.utils.to_categorical(atlas_seg4)


 #gpu = '/gpu:' + str(gpu_id)
 os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
 config = tf.ConfigProto()
 config.gpu_options.allow_growth = True
 config.allow_soft_placement = True
 set_session(tf.Session(config=config))

 # load weights of model
 with tf.device(gpu):
    net = networks.unet(vol_size, nf_enc, nf_dec)
    net.load_weights('/home/ys895/MAS4_Models/'+str(iter_num)+'.h5')
    #net.load_weights('../models/' + model_name + '/' + str(iter_num) + '.h5')

 xx = np.arange(vol_size[1])
 yy = np.arange(vol_size[0])
 zz = np.arange(vol_size[2])
 grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4) # (160, 192, 224, 3)
 #X_vol, X_seg = datagenerators.load_example_by_name('../data/test_vol.npz', '../data/test_seg.npz')
 X_vol1, X_seg1 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/981216_vc681.npz',
                                                      '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/981216_vc681.npz')

 X_vol2, X_seg2 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/990205_vc783.npz',
                                                      '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/990205_vc783.npz')

 X_vol3, X_seg3 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/990525_vc1024.npz',
                                                      '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/990525_vc1024.npz')

 X_vol4, X_seg4 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/991025_vc1379.npz',
                                                      '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/991025_vc1379.npz')

 X_vol5, X_seg5 = datagenerators.load_example_by_name('/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/991122_vc1463.npz',
                                                     '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/991122_vc1463.npz')

 # change the direction of the atlas data and volume data
 # pred[0].shape (1, 160, 192, 224, 1)
 # pred[1].shape (1, 160, 192, 224, 3)
 # X1
 with tf.device(gpu):
    pred1 = net.predict([atlas_vol1, X_vol1])
    pred2 = net.predict([atlas_vol2, X_vol1])
    pred3 = net.predict([atlas_vol3, X_vol1])
    pred4 = net.predict([atlas_vol4, X_vol1])
    #pred5 = net.predict([atlas_vol5, X_vol1])
 # Warp segments with flow
 flow1 = pred1[1][0, :, :, :, :]# (1, 160, 192, 224, 3)
 flow2 = pred2[1][0, :, :, :, :]
 flow3 = pred3[1][0, :, :, :, :]
 flow4 = pred4[1][0, :, :, :, :]
 #flow5 = pred5[1][0, :, :, :, :]

 sample1 = flow1+grid
 sample1 = np.stack((sample1[:, :, :, 1], sample1[:, :, :, 0], sample1[:, :, :, 2]), 3)
 sample2 = flow2+grid
 sample2 = np.stack((sample2[:, :, :, 1], sample2[:, :, :, 0], sample2[:, :, :, 2]), 3)
 sample3 = flow3+grid
 sample3 = np.stack((sample3[:, :, :, 1], sample3[:, :, :, 0], sample3[:, :, :, 2]), 3)
 sample4 = flow4+grid
 sample4 = np.stack((sample4[:, :, :, 1], sample4[:, :, :, 0], sample4[:, :, :, 2]), 3)
 #sample5 = flow5+grid
 #sample5 = np.stack((sample5[:, :, :, 1], sample5[:, :, :, 0], sample5[:, :, :, 2]), 3)

 warp_seg1 = interpn((yy, xx, zz), atlas_seg1[:, :, :, :], sample1, method='linear', bounds_error=False, fill_value=0) # (160, 192, 224)
 warp_seg2 = interpn((yy, xx, zz), atlas_seg2[:, :, :, :], sample2, method='linear', bounds_error=False, fill_value=0)
 warp_seg3 = interpn((yy, xx, zz), atlas_seg3[:, :, :, :], sample3, method='linear', bounds_error=False, fill_value=0)
 warp_seg4 = interpn((yy, xx, zz), atlas_seg4[:, :, :, :], sample4, method='linear', bounds_error=False, fill_value=0)
 #warp_seg5 = interpn((yy, xx, zz), atlas_seg5[:, :, :], sample5, method='nearest', bounds_error=False, fill_value=0)


 # label fusion: get the final warp_seg
 warp_seg = np.empty((160, 192, 224, atlas_seg1.shape[3]))
 warp_seg = (warp_seg1 + warp_seg2 + warp_seg3 + warp_seg4) / 4
 warp_seg = np.argmax(warp_seg, axis=3)

 vals, _ = dice(warp_seg, X_seg1[0, :, :, :, 0], labels=labels, nargout=2)
 mean1 = np.mean(vals)

 # X2
 with tf.device(gpu):
    pred1 = net.predict([atlas_vol1, X_vol2])
    pred2 = net.predict([atlas_vol2, X_vol2])
    pred3 = net.predict([atlas_vol3, X_vol2])
    pred4 = net.predict([atlas_vol4, X_vol2])
    #pred5 = net.predict([atlas_vol5, X_vol2])
 # Warp segments with flow
 flow1 = pred1[1][0, :, :, :, :]# (1, 160, 192, 224, 3)
 flow2 = pred2[1][0, :, :, :, :]
 flow3 = pred3[1][0, :, :, :, :]
 flow4 = pred4[1][0, :, :, :, :]
 #flow5 = pred5[1][0, :, :, :, :]

 sample1 = flow1+grid
 sample1 = np.stack((sample1[:, :, :, 1], sample1[:, :, :, 0], sample1[:, :, :, 2]), 3)
 sample2 = flow2+grid
 sample2 = np.stack((sample2[:, :, :, 1], sample2[:, :, :, 0], sample2[:, :, :, 2]), 3)
 sample3 = flow3+grid
 sample3 = np.stack((sample3[:, :, :, 1], sample3[:, :, :, 0], sample3[:, :, :, 2]), 3)
 sample4 = flow4+grid
 sample4 = np.stack((sample4[:, :, :, 1], sample4[:, :, :, 0], sample4[:, :, :, 2]), 3)
 #sample5 = flow5+grid
 #sample5 = np.stack((sample5[:, :, :, 1], sample5[:, :, :, 0], sample5[:, :, :, 2]), 3)

 warp_seg1 = interpn((yy, xx, zz), atlas_seg1[:, :, :, :], sample1, method='linear', bounds_error=False, fill_value=0) # (160, 192, 224)
 warp_seg2 = interpn((yy, xx, zz), atlas_seg2[:, :, :, :], sample2, method='linear', bounds_error=False, fill_value=0)
 warp_seg3 = interpn((yy, xx, zz), atlas_seg3[:, :, :, :], sample3, method='linear', bounds_error=False, fill_value=0)
 warp_seg4 = interpn((yy, xx, zz), atlas_seg4[:, :, :, :], sample4, method='linear', bounds_error=False, fill_value=0)
 #warp_seg5 = interpn((yy, xx, zz), atlas_seg5[:, :, :], sample5, method='nearest', bounds_error=False, fill_value=0)


 # label fusion: get the final warp_seg
 warp_seg = np.empty((160, 192, 224, atlas_seg1.shape[3]))
 warp_seg = (warp_seg1 + warp_seg2 + warp_seg3 + warp_seg4) / 4
 warp_seg = np.argmax(warp_seg, axis=3)

 vals, _ = dice(warp_seg, X_seg2[0,:,:,:,0], labels=labels, nargout=2)
 mean2 = np.mean(vals)
 #print(np.mean(vals), np.std(vals))

 # X3
 with tf.device(gpu):
    pred1 = net.predict([atlas_vol1, X_vol3])
    pred2 = net.predict([atlas_vol2, X_vol3])
    pred3 = net.predict([atlas_vol3, X_vol3])
    pred4 = net.predict([atlas_vol4, X_vol3])
    #pred5 = net.predict([atlas_vol5, X_vol3])
 # Warp segments with flow
 flow1 = pred1[1][0, :, :, :, :]# (1, 160, 192, 224, 3)
 flow2 = pred2[1][0, :, :, :, :]
 flow3 = pred3[1][0, :, :, :, :]
 flow4 = pred4[1][0, :, :, :, :]
 #flow5 = pred5[1][0, :, :, :, :]

 sample1 = flow1+grid
 sample1 = np.stack((sample1[:, :, :, 1], sample1[:, :, :, 0], sample1[:, :, :, 2]), 3)
 sample2 = flow2+grid
 sample2 = np.stack((sample2[:, :, :, 1], sample2[:, :, :, 0], sample2[:, :, :, 2]), 3)
 sample3 = flow3+grid
 sample3 = np.stack((sample3[:, :, :, 1], sample3[:, :, :, 0], sample3[:, :, :, 2]), 3)
 sample4 = flow4+grid
 sample4 = np.stack((sample4[:, :, :, 1], sample4[:, :, :, 0], sample4[:, :, :, 2]), 3)
 #sample5 = flow5+grid
 #sample5 = np.stack((sample5[:, :, :, 1], sample5[:, :, :, 0], sample5[:, :, :, 2]), 3)

 warp_seg1 = interpn((yy, xx, zz), atlas_seg1[:, :, :, :], sample1, method='linear', bounds_error=False, fill_value=0) # (160, 192, 224)
 warp_seg2 = interpn((yy, xx, zz), atlas_seg2[:, :, :, :], sample2, method='linear', bounds_error=False, fill_value=0)
 warp_seg3 = interpn((yy, xx, zz), atlas_seg3[:, :, :, :], sample3, method='linear', bounds_error=False, fill_value=0)
 warp_seg4 = interpn((yy, xx, zz), atlas_seg4[:, :, :, :], sample4, method='linear', bounds_error=False, fill_value=0)
 #warp_seg5 = interpn((yy, xx, zz), atlas_seg5[:, :, :], sample5, method='nearest', bounds_error=False, fill_value=0)


 # label fusion: get the final warp_seg
 warp_seg = np.empty((160, 192, 224, atlas_seg1.shape[3]))
 warp_seg = (warp_seg1 + warp_seg2 + warp_seg3 + warp_seg4) / 4
 warp_seg = np.argmax(warp_seg, axis=3)

 vals, _ = dice(warp_seg, X_seg3[0, :, :, :, 0], labels=labels, nargout=2)
 mean3 = np.mean(vals)

 # X4
 with tf.device(gpu):
    pred1 = net.predict([atlas_vol1, X_vol4])
    pred2 = net.predict([atlas_vol2, X_vol4])
    pred3 = net.predict([atlas_vol3, X_vol4])
    pred4 = net.predict([atlas_vol4, X_vol4])
    #pred5 = net.predict([atlas_vol5, X_vol4])
 # Warp segments with flow
 flow1 = pred1[1][0, :, :, :, :]# (1, 160, 192, 224, 3)
 flow2 = pred2[1][0, :, :, :, :]
 flow3 = pred3[1][0, :, :, :, :]
 flow4 = pred4[1][0, :, :, :, :]
 #flow5 = pred5[1][0, :, :, :, :]

 sample1 = flow1+grid
 sample1 = np.stack((sample1[:, :, :, 1], sample1[:, :, :, 0], sample1[:, :, :, 2]), 3)
 sample2 = flow2+grid
 sample2 = np.stack((sample2[:, :, :, 1], sample2[:, :, :, 0], sample2[:, :, :, 2]), 3)
 sample3 = flow3+grid
 sample3 = np.stack((sample3[:, :, :, 1], sample3[:, :, :, 0], sample3[:, :, :, 2]), 3)
 sample4 = flow4+grid
 sample4 = np.stack((sample4[:, :, :, 1], sample4[:, :, :, 0], sample4[:, :, :, 2]), 3)
 #sample5 = flow5+grid
 #sample5 = np.stack((sample5[:, :, :, 1], sample5[:, :, :, 0], sample5[:, :, :, 2]), 3)

 warp_seg1 = interpn((yy, xx, zz), atlas_seg1[:, :, :, :], sample1, method='linear', bounds_error=False, fill_value=0) # (160, 192, 224)
 warp_seg2 = interpn((yy, xx, zz), atlas_seg2[:, :, :, :], sample2, method='linear', bounds_error=False, fill_value=0)
 warp_seg3 = interpn((yy, xx, zz), atlas_seg3[:, :, :, :], sample3, method='linear', bounds_error=False, fill_value=0)
 warp_seg4 = interpn((yy, xx, zz), atlas_seg4[:, :, :, :], sample4, method='linear', bounds_error=False, fill_value=0)
 #warp_seg5 = interpn((yy, xx, zz), atlas_seg5[:, :, :], sample5, method='nearest', bounds_error=False, fill_value=0)


 # label fusion: get the final warp_seg
 warp_seg = np.empty((160, 192, 224, atlas_seg1.shape[3]))
 warp_seg = (warp_seg1 + warp_seg2 + warp_seg3 + warp_seg4) / 4
 warp_seg = np.argmax(warp_seg, axis=3)

 vals, _ = dice(warp_seg, X_seg4[0, :, :, :, 0], labels=labels, nargout=2)
 mean4 = np.mean(vals)

# X5
 with tf.device(gpu):
    pred1 = net.predict([atlas_vol1, X_vol5])
    pred2 = net.predict([atlas_vol2, X_vol5])
    pred3 = net.predict([atlas_vol3, X_vol5])
    pred4 = net.predict([atlas_vol4, X_vol5])
    #pred5 = net.predict([atlas_vol5, X_vol1])
 # Warp segments with flow
 flow1 = pred1[1][0, :, :, :, :]# (1, 160, 192, 224, 3)
 flow2 = pred2[1][0, :, :, :, :]
 flow3 = pred3[1][0, :, :, :, :]
 flow4 = pred4[1][0, :, :, :, :]
 #flow5 = pred5[1][0, :, :, :, :]

 sample1 = flow1+grid
 sample1 = np.stack((sample1[:, :, :, 1], sample1[:, :, :, 0], sample1[:, :, :, 2]), 3)
 sample2 = flow2+grid
 sample2 = np.stack((sample2[:, :, :, 1], sample2[:, :, :, 0], sample2[:, :, :, 2]), 3)
 sample3 = flow3+grid
 sample3 = np.stack((sample3[:, :, :, 1], sample3[:, :, :, 0], sample3[:, :, :, 2]), 3)
 sample4 = flow4+grid
 sample4 = np.stack((sample4[:, :, :, 1], sample4[:, :, :, 0], sample4[:, :, :, 2]), 3)
 #sample5 = flow5+grid
 #sample5 = np.stack((sample5[:, :, :, 1], sample5[:, :, :, 0], sample5[:, :, :, 2]), 3)

 warp_seg1 = interpn((yy, xx, zz), atlas_seg1[:, :, :, :], sample1, method='linear', bounds_error=False, fill_value=0) # (160, 192, 224)
 warp_seg2 = interpn((yy, xx, zz), atlas_seg2[:, :, :, :], sample2, method='linear', bounds_error=False, fill_value=0)
 warp_seg3 = interpn((yy, xx, zz), atlas_seg3[:, :, :, :], sample3, method='linear', bounds_error=False, fill_value=0)
 warp_seg4 = interpn((yy, xx, zz), atlas_seg4[:, :, :, :], sample4, method='linear', bounds_error=False, fill_value=0)
 #warp_seg5 = interpn((yy, xx, zz), atlas_seg5[:, :, :], sample5, method='nearest', bounds_error=False, fill_value=0)


 # label fusion: get the final warp_seg
 warp_seg = np.empty((160, 192, 224, atlas_seg1.shape[3]))
 warp_seg = (warp_seg1 + warp_seg2 + warp_seg3 + warp_seg4) / 4
 warp_seg = np.argmax(warp_seg, axis=3)

 vals, _ = dice(warp_seg, X_seg5[0, :, :, :, 0], labels=labels, nargout=2)
 mean5 = np.mean(vals)

 # compute mean of dice score
 sum = mean1 + mean2 + mean3 + mean4 + mean5
 mean_dice = sum/5
 print(mean_dice)

 # plot the outcome of warp seg
 #warp_seg = warp_seg.reshape((warp_seg.shape[1], warp_seg.shape[2], warp_seg.shape[0]))
 #warp_seg2 = np.empty(shape = (warp_seg.shape[1], warp_seg.shape[2], warp_seg.shape[0]))
 #for i in range(0,warp_seg.shape[1]):
 # warp_seg2[i,:,:] = np.transpose(warp_seg[:,i,:])
 #nu.plot.slices(warp_seg)


if __name__ == "__main__":
    #result_list = np.empty((1000,1))
    #for i in range(0,35):
    #    iterr = (i+1)*200
    #    result_list[i,0] = test(iterr,sys.argv[1])[0]
    #print(result_list)
	test(sys.argv[1], sys.argv[2])