# -*- coding: utf-8 -*-
import numpy as np
import h5py
from os import listdir
from os.path import isfile, join
from os.path import abspath
import argparse
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical
from Model import buildModel_U_net
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, CSVLogger
import scipy.io as sio
from keras.utils import to_categorical


def step_decay(epoch):
    step = 16
    num =  epoch // step
    if num % 3 == 0:
        lrate = 1e-3
    elif num % 3 == 1:
        lrate = 1e-4
    else:
        lrate = 1e-5
        # lrate = initial_lrate * 1/(1 + decay * (epoch - num * step))
    print('Learning rate for epoch {} is {}.'.format(epoch+1, lrate))
    return np.float(lrate)


def load_data(path, pattern):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    mat_list = [i for i in onlyfiles if pattern in i]

    filename = path + mat_list[0]
    matfile = h5py.File(filename, 'r')
    locs = np.array(matfile['locs'])  # input
    imps = np.array(matfile['im_p'])  # output
    maps = np.array(matfile['maps'])  # output
    nexs = np.array(matfile['nexs'])[0]  # output
    shape = np.array(matfile['label'])[0]  # output

    for i in range(1, len(mat_list)): # joining data from multiple files for matlab memory issues
        filename = path + mat_list[i]
        matfile = h5py.File(filename, 'r')
        locs_part = np.array(matfile['locs'])
        imps_part = np.array(matfile['im_p'])
        maps_part = np.array(matfile['maps'])
        nexs_part = np.array(matfile['nexs'])[0]
        shape_part = np.array(matfile['label'])[0]
        locs = np.concatenate((locs, locs_part))
        imps = np.concatenate((imps, imps_part))
        maps = np.concatenate((maps, maps_part))
        nexs = np.concatenate((nexs, nexs_part))
        shape = np.concatenate((shape, shape_part))

    return locs, imps, maps, nexs, shape


def data_normalization(locs, imps, maps, nexs, shape):
    # ===================== Locs normalization ==========================
    final_mean_locs = []
    final_std_locs = []
    for i in range(0, locs.shape[0]):
        final_mean_locs.append(np.mean(locs[i, :, :]))
        final_std_locs.append(np.std(locs[i, :, :]))
    mean_locs = np.mean(final_mean_locs)
    std_locs = np.mean(final_std_locs)

    new_locs = np.empty((locs.shape[0], locs.shape[1], locs.shape[2], 1))
    new_locs[:, :, :, 0] = (locs - mean_locs) / std_locs

    # ===================== im_p and map normalization ==========================
    final_mean_imps = []
    final_std_imps = []
    for i in range(0, imps.shape[0]):
        final_mean_imps.append(np.mean(imps[i, :, :]))
        final_std_imps.append(np.std(imps[i, :, :]))
    mean_imps = np.mean(final_mean_imps)
    std_imps = np.mean(final_std_imps)

    new_imps = np.empty((imps.shape[0], imps.shape[1], imps.shape[2], 2))
    new_imps[:, :, :, 0] = (imps - mean_imps) / std_imps
    new_imps[:, :, :, 1] = (maps / 0.5) - 1.

    new_shape = to_categorical(shape)
    new_nexs = (nexs - np.min(nexs)) / (0.5 * (np.max(nexs) - np.min(nexs))) - 1

    mdict = {"mean_locs": mean_locs, "std_locs": std_locs, "mean_imps": mean_imps, "std_imps": std_imps}

    return new_locs, new_imps, new_shape, new_nexs, mdict


def myGenerator(locs, imps, shape, nexs, batch_size):

    while True:  # generators for keras must be infinite
        n_batches = int(locs.shape[0] / batch_size)

        for i in range(n_batches - 1):
            batch = np.array(range(i * batch_size, (i + 1) * batch_size))
            new_locs = locs[batch]
            new_imps = imps[batch]
            new_shape = shape[batch]
            new_nexs = nexs[batch]

            yield new_locs, {"pred": new_imps, "activation_26": new_shape, "dense_4": new_nexs}


def train_(weights_name):
    # for reproducibility
    np.random.seed(123)

    # Load training data
    locs, imps, maps, nexs, shape = load_data("Data_unet/", "Train_unet_imps424_part")
    print("Number of training images:", locs.shape[0])

    # Get dataset dimensions
    (K, M, N) = locs.shape
    print("Shape: ", locs.shape)

    new_locs, new_imps, new_shape, new_nexs, mdict = data_normalization(locs, imps, maps, nexs, shape)

    # Saving loss at the end of each epoch
    csv_logger = CSVLogger('unet_model/training.log')

    print('-'*30)
    print('Creating and compiling the fully convolutional regression networks.')
    print('-'*30)

    model = buildModel_U_net(input_dim=(M, N, 1))
    # model.load_weights("weights_1chan_2chan_numbers_4.h5", by_name=True)
    model_checkpoint = ModelCheckpoint(filepath=weights_name, monitor='loss', save_best_only=True)
    model.summary()
    print("weights:", len(model.weights))
    print("trainable_weights:", len(model.trainable_weights))
    print("non_trainable_weights:", len(model.non_trainable_weights))
    print('...Fitting model...')
    print('-'*30)
    change_lr = LearningRateScheduler(step_decay)

    # Start image generator
    batch_size = 16
    gen = myGenerator(new_locs, new_imps, new_shape, new_nexs, batch_size=batch_size)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(gen,
                        steps_per_epoch=new_locs.shape[0]//batch_size,
                        # samples_per_epoch=new_locs.shape[0],
                        nb_epoch=192,
                        callbacks=[model_checkpoint, change_lr, csv_logger])

    # mdict = {"mean_locs": mean_locs, "std_locs": std_locs, "mean_imps": mean_imps, "std_imps": std_imps}
    sio.savemat("unet_model/Training_dict.mat", mdict)


if __name__ == '__main__':
    # start a parser
    parser = argparse.ArgumentParser()

    # path of the training data: patches and heatmaps, created in MATLAB using
    # the function "GenerateTrainingExamples.m"
    # parser.add_argument('--filename', type=str, help="path to generated training data m-file")

    # path for saving the optimal model weights and normalization factors after
    # training with the function "train_model.py" is completed.
    parser.add_argument('--weights_name', type=str, help="path to save model weights as hdf5-file")
    # parser.add_argument('--meanstd_name', type=str, help="path to save normalization factors as m-file")
    # parser.add_argument('--accloss_name', type=str, help="path to save accuracy and loss as m-file")

    # parse the input arguments
    args = parser.parse_args()

    # run the training process
    train_(abspath(args.weights_name))
