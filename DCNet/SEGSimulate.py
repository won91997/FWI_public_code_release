
import os

import numpy as np
from skimage.measure import block_reduce
import scipy.io


def SEGSimulate(base_dir, start, instance_size, in_channels, split, data_dsp_blk, label_dsp_blk, model_dim):
    if split == 'train':

        base_dir = 'D:'
        _data_set = np.load(os.path.join(base_dir, 'georecData.npy'))
        _label_set = np.load(os.path.join(base_dir, 'vmodelData.npy'))
        return _data_set[:instance_size], _label_set[0][:instance_size]

    if split == 'train':
        data_path = os.path.join(base_dir, 'train_data/georec_train')
        label_path = os.path.join(base_dir, 'train_data/vmodel_train')
    else:
        data_path = os.path.join(base_dir, 'test_data/georec_train')
        label_path = os.path.join(base_dir, 'test_data/vmodel_train')

    for i in range(start, start + instance_size):
        data_set_ = scipy.io.loadmat(data_path+'/georec'+str(i+1))['data']
        label_set_ = scipy.io.loadmat(label_path+'/vmodel'+str(i+1))['vmodel']
        if (i+1) % 50 == 0 : print(data_path+'/georec'+str(i+1))
        # Change the dimention [h, w, c] --> [c, h, w]
        for k in range(0, in_channels):
            data1_set = np.float32(data_set_[:, :, k])
            data1_set = np.float32(data1_set)
            # Data downsampling
            # note that the len(data11_set.shape)=len(block_size.shape)=2
            data1_set = block_reduce(data1_set, block_size=data_dsp_blk, func=decimate)
            data_dsp_dim = data1_set.shape
            data1_set = data1_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
            if k == 0:
                train1_set = data1_set
            else:
                train1_set = np.append(train1_set, data1_set, axis=0)

        data2_set = np.float32(label_set_).reshape(model_dim)
        # Label downsampling
        data2_set = block_reduce(data2_set, block_size=label_dsp_blk, func=np.max)
        label_dsp_dim = data2_set.shape
        data2_set = data2_set.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
        data2_set = np.float32(data2_set)
        if i == start:
            train_set = train1_set
            label_set = data2_set
        else:
            train_set = np.append(train_set, train1_set, axis=0)
            label_set = np.append(label_set, data2_set, axis=0)

    train_set = train_set.reshape((instance_size, in_channels, data_dsp_dim[0] * data_dsp_dim[1]))
    label_set = label_set.reshape((instance_size, 1, label_dsp_dim[0] * label_dsp_dim[1]))

    return train_set, label_set, data_dsp_dim, label_dsp_dim


# downsampling function by taking the middle value
def decimate(a, axis):
    idx = np.round((np.array(a.shape)[np.array(axis).reshape(1, -1)] + 1.0) / 2.0 - 1).reshape(-1)
    downa = np.array(a)[:, :, idx[0].astype(int), idx[1].astype(int)]
    return downa
