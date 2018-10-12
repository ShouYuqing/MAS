"""
single atlas segmentation based on Voxelmorph and Neuron
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model

# project imports
import datagenerators
import networks
import losses