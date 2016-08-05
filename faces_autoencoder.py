import os
import re
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

IM_WIDTH = 16
IM_HEIGHT = 16
IM_AREA = IM_WIDTH * IM_HEIGHT

IMAGES_PATH = "/Users/mac/Documents/Programmation/Intelligence artificielle/Computer_Vision/Projets/pybrain_simple_face_detection/face_detection/faces/train/face/"

def load_image_vector(img_path):
    try:
        M = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    except:
        read_pgm(img_path)
    M = cv2.resize(M, (IM_WIDTH, IM_HEIGHT))
    M = M.reshape((1, IM_AREA))
    return M[0]

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                         count=int(width)*int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))

def save_network_state(signum, stack):
    file_name = "autoencoder_100px_300faces_{}{}".format(int(time()), randrange(0, 10))
    f = open(file_name, "wb")
    pickle.dump(network, f)
    f.close()
    print "network saved as : {}".format(file_name)


network = buildNetwork(IM_AREA,
                       IM_AREA * 2 // 3,
                       IM_AREA)

#preload all images
images = []

for img_name in os.listdir(IMAGES_PATH):
    try:
        vector = load_image_vector(IMAGES_PATH + img_name)
        if vector != None and len(vector) == IM_AREA:
            images.append(vector)
    except:
        continue

print "dataset with {} elements done...".format(len(images))

signal.signal(signal.SIGUSR1, save_network_state)

#stochastic training
for i in range(len(images) // 2):
    img = choice(images)
    dataset = SupervisedDataSet(IM_AREA, IM_AREA)
    dataset.addSample(img, img)
    trainer = BackpropTrainer(network, dataset)
    trainer.train()

print "training completed... (1 pass)"

save_network_state(0, 0)

print "done !! :D"

