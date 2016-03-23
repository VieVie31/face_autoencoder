import os
import cv2
import pickle
import signal
import numpy as np
import matplotlib.pyplot as plt

from time import *
from random import *
from pybrain.structure import FullConnection
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules.biasunit import BiasUnit
from pybrain.datasets.classification import SupervisedDataSet
from pybrain.structure.modules import SigmoidLayer, LinearLayer, SoftmaxLayer

#using auto encoder

AUTOENCODER_PATH = "autoencoder_100px_300faces_14587588881"

IM_WIDTH = 10
IM_HEIGHT = 10
IM_AREA = IM_WIDTH * IM_HEIGHT


def load_network():
    f = open(AUTOENCODER_PATH, "rb")
    network = pickle.load(f)
    f.close()
    return network


network = load_network()

face_0 = network.activate([0] * IM_AREA).reshape((IM_WIDTH, IM_HEIGHT))
plt.imshow(face_0)
plt.show()

face_255 = network.activate([255] * IM_AREA).reshape((IM_WIDTH, IM_HEIGHT))
plt.imshow(face_255)
plt.show()

while True:
    #yes it's cool to see how the auto encoder will transform
    #random noise images into a portrait shape... :D
    face_random = network.activate([randrange(0, 256) for i in range(IM_AREA)]).reshape((IM_WIDTH, IM_HEIGHT))
    plt.imshow(face_random)
    plt.show()




