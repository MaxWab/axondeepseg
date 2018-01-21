# -*- coding: utf-8 -*-


import time
import math
import numpy as np
import os
import tensorflow as tf
import pickle
from data_management.input_data import input_data
from config_tools import generate_config

# Imports
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # To get rid of Tensorflow warnings.

# First, we define the functions that we are going to use to construct the network.
# Most functions are encapsulations of low level layers to enable clear and easy call in the main network building
# function.

# ------ LAYERS

# Convolution

def atrous_conv_relu(x, n_out_chan, k_size, k_stride, scope, dilation_rate=1,
              w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              training_phase=True, activate_bn = True, bn_decay = 0.999, keep_prob=1.0):
    '''
    Default data format is NHWC.
    Initializers for weights and bias are already defined (default).
    :param training_phase: Whether we are in the training phase (True) or testing phase (False)
    '''
    
    with tf.variable_scope(scope):
        if activate_bn == True:
            net = tf.contrib.layers.conv2d(x, num_outputs=n_out_chan, kernel_size=k_size, stride=k_stride, 
                                       rate=dilation_rate,
                                       activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'scale':True, 'is_training':training_phase,
                                                          'decay':bn_decay,'scope':'bn'},
                                       weights_initializer = w_initializer, scope='convolution'
                                      )
        else:
            net = tf.contrib.layers.conv2d(x, num_outputs=n_out_chan, kernel_size=k_size, stride=k_stride, 
                                       rate=dilation_rate,
                                       activation_fn=tf.nn.relu, weights_initializer = w_initializer, scope='convolution'
                                      )
        net = tf.contrib.layers.dropout(net, keep_prob=keep_prob, is_training=training_phase)
        tf.add_to_collection('activations',net)
        return net

def conv_relu(x, n_out_chan, k_size, k_stride, scope, 
              w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              training_phase=True, activate_bn = True, bn_decay = 0.999, keep_prob=1.0):
    '''
    Default data format is NHWC.
    Initializers for weights and bias are already defined (default).
    :param training_phase: Whether we are in the training phase (True) or testing phase (False)
    '''
    
    with tf.variable_scope(scope):
        if activate_bn == True:
            net = tf.contrib.layers.conv2d(x, num_outputs=n_out_chan, kernel_size=k_size, stride=k_stride, 
                                       activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
                                       normalizer_params={'scale':True, 'is_training':training_phase,
                                                          'decay':bn_decay,'scope':'bn'},
                                       weights_initializer = w_initializer, scope='convolution'
                                      )
        else:
            net = tf.contrib.layers.conv2d(x, num_outputs=n_out_chan, kernel_size=k_size, stride=k_stride, 
                                       activation_fn=tf.nn.relu, weights_initializer = w_initializer, scope='convolution'
                                      )
        net = tf.contrib.layers.dropout(net, keep_prob=keep_prob, is_training=training_phase)
        tf.add_to_collection('activations',net)
        return net
 
    
    

def maxpool(x, k_size, k_stride, scope, padding='VALID'):
    return tf.contrib.layers.max_pool2d(x,k_size,stride=k_stride,padding=padding,scope=scope)


# ------------------------ NETWORK STRUCTURE


def uconv_net(x, training_config, phase, bn_updated_decay = None, verbose = True):
    """
    Create the U-net.
    Input :
        x : TF object to define
        config : dict : described in the header.
        image_size : int : The image size

    Output :
        The U-net.
    """
    
    # Load the variables
    image_size = training_config["trainingset_patchsize"]
    n_classes = training_config["n_classes"]
    depth = training_config["depth"]
    dropout = training_config["dropout"]
    number_of_convolutions_per_layer = training_config["convolution_per_layer"]
    size_of_convolutions_per_layer = training_config["size_of_convolutions_per_layer"]
    features_per_convolution = training_config["features_per_convolution"]
    downsampling = training_config["downsampling"]
    activate_bn = training_config["batch_norm_activate"]
    dilation_rate = training_config["dilation_rate"]
    if bn_updated_decay is None:
        bn_decay = training_config["batch_norm_decay_starting_decay"]
    else:
        bn_decay = bn_updated_decay

    # Input picture shape is [batch_size, height, width, number_channels_in] (number_channels_in = 1 for the input layer)
    net = tf.reshape(x, shape=[-1, image_size, image_size, 1])
    data_temp = x
    data_temp_size = [image_size]
    relu_results = []

    ####################################################################
    ######################### CONTRACTION PHASE ########################
    ####################################################################
    
    # Two first convolutions, dilation rate 1
    conv1 = atrous_conv_relu(net, 32, 3, k_stride=1, dilation_rate=1,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv1')
    
    conv2 = atrous_conv_relu(conv1, 32, 3, k_stride=1, dilation_rate=1,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv2')    
    
    # Next two convolutions, dilation rate 2
    conv3 = atrous_conv_relu(conv2, 32, 3, k_stride=1, dilation_rate=2,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv3')
    
    conv4 = atrous_conv_relu(conv3, 32, 3, k_stride=1, dilation_rate=2,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv4')
    
    # Next, the branches for ASPP
    
    #### Branch (a), 1x1, no dilation rate
    
    # Convo 3x3
    conv5a = atrous_conv_relu(conv4, 32, 3, k_stride=1, dilation_rate=1,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv5a')
    
    # Convo 1x1
    conv6a = atrous_conv_relu(conv5a, 32, 1, k_stride=1, dilation_rate=1,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv6a')

    #### Branch (b), 3x3, dilation rate = 6
    
    # Convo 3x3
    conv5b = atrous_conv_relu(conv4, 32, 3, k_stride=1, dilation_rate=6,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv5b')
    
    # Convo 3x3
    conv6b = atrous_conv_relu(conv5b, 32, 3, k_stride=1, dilation_rate=6,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv6b')

    #### Branch (c), 3x3, dilation rate = 12
    
    # Convo 3x3
    conv5c = atrous_conv_relu(conv4, 32, 3, k_stride=1, dilation_rate=12,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv5c')
    
    # Convo 3x3
    conv6c = atrous_conv_relu(conv5c, 32, 3, k_stride=1, dilation_rate=12,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv6c')


    #### Branch (d), 3x3, dilation rate = 18
    
    # Convo 3x3
    conv5d = atrous_conv_relu(conv4, 32, 3, k_stride=1, dilation_rate=18,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv5d')
    
    # Convo 3x3
    conv6d = atrous_conv_relu(conv5d, 32, 3, k_stride=1, dilation_rate=18,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv6d')

    #### Branch (e), 3x3, dilation rate = 24
    
    # Convo 3x3
    conv5e = atrous_conv_relu(conv4, 32, 3, k_stride=1, dilation_rate=24,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv5e')
    
    # Convo 3x3
    conv6e = atrous_conv_relu(conv5e, 32, 3, k_stride=1, dilation_rate=24,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv6e')
    
    # Global Average Pooling, used for the global context
    #conv6f = tf.reduce_mean(conv2, [1, 2])
    #conv6f = tf.expand_dims(conv6f, 1)
    #conv6f = tf.tile(conv6f,[1, image_size*image_size, 1],'rep_mean')
    #conv6f = tf.reshape(tensor=conv6f, shape=[-1, image_size,image_size, 32],name='reshape_mean')
    
    # Concatenation step
#    concat = tf.concat([conv6a, conv6b, conv6c, conv6d, conv6f, conv6e], axis=3)
    concat = tf.concat([conv6a, conv6b, conv6c, conv6d, conv6e], axis=3)    
    concat = tf.contrib.layers.batch_norm(concat, scale=True, is_training=phase,decay=bn_decay,scope='bn')
    concat = tf.contrib.layers.dropout(concat, keep_prob=dropout, is_training=phase)
    
    # Last convolution, amort step
    conv7 = atrous_conv_relu(concat, 64, 1, k_stride=1, dilation_rate=1,
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            training_phase=phase, activate_bn=activate_bn, bn_decay = bn_decay,
                            keep_prob=dropout, scope='conv7')
    
    finalconv = tf.contrib.layers.conv2d(conv7, num_outputs=n_classes, kernel_size=1, stride=1, scope='finalconv', padding='SAME')

    # Adding summary of the activations for the last convolution
    tf.summary.histogram('activations_last', finalconv)
       
    # Finally we compute the activations of the last layer
    final_result = tf.reshape(finalconv,
        [tf.shape(finalconv)[0], image_size*image_size, n_classes])

    return final_result