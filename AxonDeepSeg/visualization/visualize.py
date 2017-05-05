import pickle
import matplotlib.pyplot as plt
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from ..testing.segmentation_scoring import score_analysis, dice
from sklearn import preprocessing
import os
from tabulate import tabulate
import numpy as np


def visualize_training(path_model, path_model_init=None, start_visu=0):
    """
    :param path_model: path of the folder with the model parameters .ckpt
    :param path_model_init: if the model is initialized by another, path of its folder
    :param start_visu: first iterations can reach extreme values, start_visu set another start than epoch 0
    :return: no return

    figure(1) represent the evolution of the loss and the accuracy evaluated on the test set along the learning process
    figure(2) if learning initialized by another, evolution of the model of initialisation and of the new are merged

    """

    file = open(path_model + '/evolution.pkl', 'r')  # training variables : loss, accuracy, epoch
    evolution = pickle.load(file)

    if path_model_init:
        file_init = open(path_model_init + '/evolution.pkl', 'r')
        evolution_init = pickle.load(file_init)
        last_epoch = evolution_init['steps'][-1]

        evolution_merged = {}  # Merging the two plots : learning of the init and learning of the model
        for key in ['steps', 'accuracy', 'loss']:
            evolution_merged[key] = evolution_init[key] + evolution[key]

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.plot(evolution_merged['steps'][start_visu:], evolution_merged['accuracy'][start_visu:], '-',
                label='accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(ymin=0.7)
        ax2 = ax.twinx()
        ax2.axvline(last_epoch, color='k', linestyle='--')
        plt.title('Evolution merged (before and after restoration')
        ax2.plot(evolution_merged['steps'][start_visu:], evolution_merged['loss'][start_visu:], '-r', label='loss')
        plt.ylabel('Loss')
        plt.ylim(ymax=100)
        plt.xlabel('Epoch')

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(evolution['steps'][start_visu:], evolution['accuracy'][start_visu:], '-', label='accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(ymin=0.7)
    ax2 = ax.twinx()
    plt.title('Accuracy and loss evolution')
    ax2.plot(evolution['steps'][start_visu:], evolution['loss'][start_visu:], '-r', label='loss')
    plt.ylabel('Loss')
    plt.ylim(ymax=100)
    plt.xlabel('Epoch')
    plt.show()


def visualize_segmentation(path):
    """
    :param path: path of the folder including the data and the results obtained after by the segmentation process.
    :return: no return
    if there is a mask (ground truth) in the folder, scores are calculated : sensitivity, errors and dice
    figure(1) segmentation without mrf
    figure(2) segmentation with mrf
    if there is MyelinSeg.jpg in the folder, myelin and image, myelin and axon segmentated, myelin and groundtruth are represented
    """

    path_img = path + '/image.jpg'
    mask = False

    if not 'results.pkl' in os.listdir(path):
        print 'results not present'

    file = open(path + '/results.pkl', 'r')
    res = pickle.load(file)

    prediction_mrf = res['prediction_mrf']
    prediction = res['prediction']
    image_init = imread(path_img, flatten=False, mode='L')
    predict = np.ma.masked_where(prediction == 0, prediction)
    predict_mrf = np.ma.masked_where(prediction_mrf == 0, prediction_mrf)

    i_figure = 1

    plt.figure(i_figure)
    plt.title('Axon Segmentation (with mrf) mask')
    plt.imshow(image_init, 'gray')
    plt.hold(True)
    plt.imshow(predict_mrf, 'hsv', alpha=0.7)

    i_figure += 1

    plt.figure(i_figure)
    plt.title('Axon Segmentation (without mrf) mask')
    plt.imshow(image_init, 'gray')
    plt.imshow(predict, 'hsv', alpha=0.7)

    i_figure += 1

    if 'mask.jpg' in os.listdir(path):
        Mask = True
        path_mask = path + '/mask.jpg'
        mask = preprocessing.binarize(imread(path_mask, flatten=False, mode='L'), threshold=125)

        acc = accuracy_score(prediction.reshape(-1, 1), mask.reshape(-1, 1))
        score = score_analysis(image_init, mask, prediction)
        Dice = dice(image_init, mask, prediction)['dice']
        Dice_mean = Dice.mean()
        acc_mrf = accuracy_score(prediction_mrf.reshape(-1, 1), mask.reshape(-1, 1))
        score_mrf = score_analysis(image_init, mask, prediction_mrf)
        Dice_mrf = dice(image_init, mask, prediction_mrf)['dice']
        Dice_mrf_mean = Dice_mrf.mean()

        headers = ["MRF", "accuracy", "sensitivity", "precision", "diffusion", "Dice"]
        table = [["False", acc, score[0], score[1], score[2], Dice_mean],
                 ["True", acc_mrf, score_mrf[0], score_mrf[1], score_mrf[2], Dice_mrf_mean]]

        subtitle2 = '\n\n---Scores---\n\n'
        scores = tabulate(table, headers)
        text = subtitle2 + scores

        subtitle3 = '\n\n---Dice Percentiles---\n\n'
        headers = ["MRF", "Dice 10th", "50th", "90th"]
        table = [["False", np.percentile(Dice, 10), np.percentile(Dice, 50), np.percentile(Dice, 90)],
                 ["True", np.percentile(Dice_mrf, 10), np.percentile(Dice_mrf, 50), np.percentile(Dice_mrf, 90)]]
        scores_2 = tabulate(table, headers)

        text = text + subtitle3 + subtitle3 + scores_2
        print text

        file = open(path + "/Report_results.txt", 'w')
        file.write(text)
        file.close()

    if 'MyelinSeg.jpg' in os.listdir(path):
        path_myelin = path + '/MyelinSeg.jpg'
        myelin = preprocessing.binarize(imread(path_myelin, flatten=False, mode='L'), threshold=125)
        myelin = np.ma.masked_where(myelin == 0, myelin)

        plt.figure(i_figure)
        plt.title('Myelin Segmentation')
        plt.imshow(image_init, 'gray')
        plt.imshow(myelin, 'hsv', alpha=0.7)

        i_figure += 1

        if Mask:
            plt.figure(i_figure)
            plt.title('Myelin - GroundTruth')
            plt.imshow(mask, cmap=plt.get_cmap('gray'))
            plt.hold(True)
            plt.imshow(myelin, alpha=0.7)

    plt.show()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--path_model", required=True, help="")
    ap.add_argument("-m_init", "--path_model_init", required=False, help="")

    args = vars(ap.parse_args())
    path_model = args["path_model"]
    path_model_init = args["path_model_init"]

    visualize_training(path_model, path_model_init)