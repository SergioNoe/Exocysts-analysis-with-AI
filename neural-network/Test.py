# Imports
import matplotlib
matplotlib.use('agg')
import numpy as np
import h5py
import time
from os import listdir
from os.path import isfile, join
from os.path import abspath
import argparse
from Model import buildModel_U_net
import scipy.io as sio
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import to_categorical


def load_data(path, pattern):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    mat_list = [i for i in onlyfiles if pattern in i]

    filename = path + mat_list[0]
    matfile = h5py.File(filename, 'r')
    locs = np.array(matfile['locs'])
    imps = np.array(matfile['im_p'])
    maps = np.array(matfile['maps'])
    nexs = np.array(matfile['nexs'])[0]
    shape = np.array(matfile['label'])[0]

    for i in range(1, len(mat_list)):
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


def test_model(weights_file):
    """
    Inputs:  weights_file: the saved weights file generated in train_model

    Outputs: mat file with predictions and labels
    """


    # Loading files
    ## Obtaining test data
    locs, imps, maps, nexs, shape = load_data("Data_unet/", "Test_unet_imps424_part")
    print("Number of testing images:", locs.shape[0])

    ## Get dataset dimensions
    (K, M, N) = locs.shape
    print("Shape: ", locs.shape)

    ## Build the model for a bigger image
    model = buildModel_U_net((M, N, 1))

    ## Load the trained weights
    model.load_weights(weights_file)

    ## load mean and std
    matfile = sio.loadmat("unet_model/Training_dict")
    mean_locs = np.array(matfile['mean_locs'])
    std_locs = np.array(matfile['std_locs'])

    new_locs = np.empty((locs.shape[0], locs.shape[1], locs.shape[2], 1))
    new_locs[:, :, :, 0] = (locs - mean_locs) / std_locs
    # new_locs[:, :, :, 1] = (locs - mean_locs) / std_locs

    # Prediction
    ## Make a prediction and time it
    start = time.time()
    predicted_density = model.predict(new_locs, batch_size=1)
    end = time.time()
    print(end - start)

    # print(type(predicted_density))
    # print(len(predicted_density))
    # print(predicted_density[0].shape)
    # print(predicted_density[1].shape)
    # print(predicted_density[2].shape)

    # print("Visualizing things")
    # print("Pred_shapes 0 to 10:")
    # print(predicted_density[1][0:10])

    shapes = to_categorical(shape)

    # Confusion matrix
    Conf_mat = confusion_matrix(np.argmax(shapes, axis=1), np.argmax(predicted_density[1], axis=1))
    print("Confusion Matrix:")
    print(Conf_mat)

    # Calculate accuracy
    acc = accuracy_score(np.argmax(shapes, axis=1), np.argmax(predicted_density[1], axis=1))
    print("Accuracy shapes: ", acc)

    # Save predictions to a matfile to open later in matlab
    mdict = {"Pred": predicted_density[0], "Pred_shapes": predicted_density[1], "Pred_nexs": predicted_density[2],
             "locs": locs, "imps": imps, "maps": maps, "nexs": nexs, "shapes": shape}
    sio.savemat("unet_model/Test_dict.mat", mdict)


if __name__ == '__main__':

    # start a parser
    parser = argparse.ArgumentParser()

    # path of the test data labels generated with matlab
    # parser.add_argument('--filename', type=str, help="path to generated test data m-file")

    # path of the optimal model weights and normalization factors, saved after
    # training with the function "train_model.py" is completed.
    parser.add_argument('--weights_name', help="path to the trained model weights as hdf5-file")
    # parser.add_argument('--meanstd_name', help="path to the saved normalization factors as m-file")

    # path for saving the Superresolution reconstruction matfile
    # parser.add_argument('--savename', type=str, help="path for saving the Superresolution reconstruction matfile")

    # parse the input arguments
    args = parser.parse_args()

    # run the testing/reconstruction process
    test_model(abspath(args.weights_name))
