# A simple portrait image autoencoder
This code is a simple autoencoder for faces build with pybrain.

##Requirements
To get pybrain : 
```bash
pip install pybrain
```

For some convenience I use also : opencv (cv2), numpy, matplotlib.
This script run on python 2.7 !!

##Running (the interesting part...)
This script (faces_autoencoder.py) will load all pictures conained in the IMAGES_PATH directory and 
will resize then into a 10 by 10 pixels image and put each row in a vector of len 100 (10x10).
We will build a supervised dataset where the input is equal to the output.

The autoencoder as one input layer of the same size of the output layer but the hidden layer 
is 2/3 smaller than the input (or output...), so the neural network must extract the main
features of each image to get the better output with some lost...

As with many images the network can take a while (a very long while...) I put a simple signal
to make some safeguard of the trained network...
You just have to start the training of th auto encoder with :
```bash
$ python faces_autoencoder.py &
PID : 2854
...
```
And to get à the moment m the state of the neural network you just have to write :
```bash
$ kill -USR1 2854
```

For learning more about autoencoders take a look [here](https://en.wikipedia.org/wiki/Autoencoder)
