import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

(X_train, _), (_, _) = mnist.load_data()