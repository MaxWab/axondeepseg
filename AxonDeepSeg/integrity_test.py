
# -*- coding: utf-8 -*-

# Basic integrity test to check is AxonDeepSeg is correctly installed
# Launches a segmentation in the data_test folder

try:

    import json
    import os
    from AxonDeepSeg.testing.segmentation_scoring import *
    from time import time

    from scipy.misc import imread, imsave

    # input parameters

    path_testing = "../AxonDeepSeg/data_test/"
    model_name = 'default_SEM_model_v1'
    path_model = '../AxonDeepSeg/models/' + model_name


    path_configfile = path_model + '/config_network.json'

    if not os.path.exists(path_model):
        os.makedirs(path_model)

    with open(path_configfile, 'r') as fd:
        config_network = json.loads(fd.read())


    from AxonDeepSeg.apply_model import axon_segmentation

    prediction = axon_segmentation([path_testing], ["image.png"], path_model, config_network,verbosity_level=0)


    mask = imread(path_testing + '/mask.png', flatten=True)
    pred = imread(path_testing + '/AxonDeepSeg.png', flatten=True)


    gt_axon = mask > 200
    gt_myelin = np.logical_and(mask >= 50, mask <= 200)

    pred_axon = pred > 200
    pred_myelin = np.logical_and(pred >= 50, pred <= 200)

    dice_axon = pw_dice(pred_axon, gt_axon)
    dice_myelin = pw_dice(pred_myelin, gt_myelin)

    print "* * * Integrity test passed. AxonDeepSeg is correctly installed. * * * "

except IOError:

    print "Integrity test failed... "
























































