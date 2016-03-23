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

print "PID : {}".format(os.getpid())

print "modules imported..."

IM_WIDTH = 10
IM_HEIGHT = 10
IM_AREA = IM_WIDTH * IM_HEIGHT

IMAGES_PATH = "/Users/mac/Desktop/Pictures/"

def load_image_vector(img_path):
    M = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    M = cv2.resize(M, (IM_WIDTH, IM_HEIGHT))
    M = M.reshape((1, IM_AREA))
    return M[0]

def save_network_state(signum, stack):
    file_name = "autoencoder_100px_300faces_{}{}".format(int(time()), randrange(0, 10))
    f = open(file_name, "wb")
    pickle.dump(network, f)
    f.close()
    print "network saved as : {}".format(file_name)


network = buildNetwork(IM_AREA, IM_AREA * 2 // 3, IM_AREA)
dataset = SupervisedDataSet(IM_AREA, IM_AREA)

for img_name in os.listdir(IMAGES_PATH):
    try:
        vector = load_image_vector(IMAGES_PATH + img_name)
        if vector != None and len(vector) == IM_AREA:
            dataset.addSample(vector, vector)
    except:
        continue

print "dataset with {} elements done...".format(dataset.getLength())

signal.signal(signal.SIGUSR1, save_network_state)

trainer = BackpropTrainer(network, dataset)

out = trainer.trainUntilConvergence()

print "training completed..."

save_network_state(0, 0)

print "done !! :D"

