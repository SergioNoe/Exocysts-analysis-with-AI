# -*- coding: utf-8 -*-
"""
Original author: Weidi Xie
"""

from __future__ import absolute_import
from __future__ import print_function
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Activation,
    Concatenate,
    Dropout,
    Reshape,
    Permute,
    Dense,
    UpSampling2D,
    Flatten,
    Add
    )
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import (
    Convolution2D)
from keras.layers.pooling import (
    MaxPooling2D,
    AveragePooling2D
    )
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import tensorflow as tf


weight_decay = 1e-5
K.set_image_dim_ordering('tf')


def _conv_bn_relu(nb_filter, row, col, subsample=(1,1)):
    def f(input):
        conv_a = Convolution2D(nb_filter, row, col, subsample=subsample,
                               init='orthogonal',
                               border_mode='same', bias=False)(input)
        # conv_a.trainable = False
        norm_a = BatchNormalization()(conv_a)
        # norm_a.trainable = False
        act_a = Activation(activation='relu')(norm_a)
        # act_a.trainable = False
        return act_a
    return f


def _conv_bn_relu_x2(nb_filter, row, col, subsample=(1, 1)):
    def f(input):
        _input = Convolution2D(nb_filter, 1, 1, subsample=subsample,
                               init='orthogonal', border_mode='same', bias=False,
                               W_regularizer=l2(weight_decay),
                               b_regularizer=l2(weight_decay))(input)
        # _input.trainable = False
        conv_a = Convolution2D(nb_filter, row, col, subsample=subsample,
                               init='orthogonal', border_mode='same', bias=False,
                               W_regularizer=l2(weight_decay),
                               b_regularizer=l2(weight_decay))(input)
        # conv_a.trainable = False
        norm_a = BatchNormalization()(conv_a)
        # norm_a.trainable = False
        act_a = Activation(activation='relu')(norm_a)
        # act_a.trainable = False
        conv_b = Convolution2D(nb_filter, row, col, subsample=subsample,
                               init='orthogonal', border_mode='same', bias=False,
                               W_regularizer=l2(weight_decay),
                               b_regularizer=l2(weight_decay))(act_a)
        # conv_b.trainable = False
        norm_b = BatchNormalization()(conv_b)
        # norm_b.trainable = False
        act_b = Activation(activation='relu')(norm_b)
        # act_b.trainable = False
        return Add()([act_b, _input])
    return f


def U_net_base(input, nb_filter=64):
    block1 = _conv_bn_relu_x2(nb_filter, 3, 3)(input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
    # pool1.trainable = False
    # =========================================================================
    block2 = _conv_bn_relu_x2(nb_filter, 3, 3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    # pool2.trainable = False
    # =========================================================================
    block3 = _conv_bn_relu_x2(nb_filter, 3, 3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    # pool3.trainable = False
    # =========================================================================
    block4 = _conv_bn_relu_x2(nb_filter, 3, 3)(pool3)
    up4 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(block4), block3])
    # up4.trainable = False
    # =========================================================================
    block5 = _conv_bn_relu_x2(nb_filter, 3, 3)(up4)
    up5 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(block5), block2])
    # up5.trainable = False
    # =========================================================================
    block6 = _conv_bn_relu_x2(nb_filter, 3, 3)(up5)
    up6 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(block6), block1])
    # up6.trainable = False
    # =========================================================================
    block7 = _conv_bn_relu(nb_filter, 3, 3)(up6)
    return block7


def conv_classifier(concatenated_layer, nb_filter=64):
    block1 = _conv_bn_relu_x2(nb_filter, 3, 3)(concatenated_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu_x2(nb_filter, 3, 3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu_x2(nb_filter, 3, 3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)

    return pool3


def dense_classifier(conv_output, neurons_last_layer=5, act_last_layer="softmax"):
    flat = Flatten()(conv_output)
    dense1 = Dense(64, activation='relu')(flat)

    if act_last_layer == "softmax":
        dense2 = Dense(neurons_last_layer)(dense1)
        activation_layer = Activation(tf.nn.softmax)(dense2)
        return activation_layer

    else:
        dense2 = Dense(neurons_last_layer, activation=act_last_layer)(dense1)
        return dense2


def buildModel_U_net(input_dim):
    input_ = Input(shape=input_dim)
    # =========================================================================
    act_ = U_net_base(input_, nb_filter=64)
    density_pred = Convolution2D(2, 1, 1, bias=False, activation='linear',
                                 init='orthogonal', name='pred', border_mode='same')(act_)
    density_pred.trainable = False

    # =========================================================================
    concat_ltc = Concatenate(axis=-1)([input_, density_pred])
    conv_result_shapes = conv_classifier(concat_ltc)
    conv_result_nexs = conv_classifier(concat_ltc)

    shapes_result = dense_classifier(conv_result_shapes)
    nexs_result = dense_classifier(conv_result_nexs, neurons_last_layer=1, act_last_layer="linear")

    # =========================================================================
    model = Model(input=input_, output=[density_pred, shapes_result, nexs_result])
    opt = Adam(lr=0.001)  # RMSprop(1e-3)
    # model.compile(optimizer=opt, loss='mse')  # MSE better than cat_cross
    model.compile(optimizer=opt,
                  loss={
                      "pred": "mse",
                      "activation_26": "categorical_crossentropy",
                      "dense_4": "mse"
                  },
                  loss_weights={
                      'pred': 100.,
                      'activation_26': 1.,
                      'dense_4': 1.}
                  )
    return model
