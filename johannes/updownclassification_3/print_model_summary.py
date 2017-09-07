#!/usr/bin/env python
# coding: utf-8

# this script uses theos datasets. That is a try to reduce overfitting

import jkutils
from jkutils import zenith_to_binary, read_files
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32" 
os.environ["PATH"] += os.pathsep + '/usr/local/cuda/bin/'
os.environ['PYTHONUNBUFFERED'] = '1'
import sys
import inspect
import numpy as np
with jkutils.suppress_stdout_stderr(): #prevents printed info from theano
    import theano
    # theano.config.device = 'gpu'
    # theano.config.floatX = 'float32'
    import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D,\
 BatchNormalization, MaxPooling2D,Convolution3D,MaxPooling3D
from keras.optimizers import SGD, Adagrad
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.constraints import maxnorm
from keras import regularizers
from sklearn.model_selection import train_test_split
import h5py
import datetime
import gc
from ConfigParser import ConfigParser
import argparse
import tables
import math
import time
import resource
import shelve
import itertools
import shutil

## constants ##
energy, azmiuth, zenith, muex = 0, 1, 2, 3

################# Function Definitions ####################################################################

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of the File containing the model", type=str, default='FCNN_v1.cfg')
    args = parser.parse_args()
    return args

def add_layer(model, layer, args, kwargs):
    eval('model.add({}(*args,**kwargs))'.format(layer))
    return

def base_model(model_def):
    model = Sequential()
    with open(model_def) as f:
        args = []
        kwargs = dict()
        layer = ''
        mode = 'args'
        for line in f:
            cur_line = line.strip()
            if cur_line == '' and layer != '':
                add_layer(model, layer, args, kwargs)
                args = []
                kwargs = dict()
                layer = ''
            elif cur_line[0] == '#':
                continue
            elif cur_line == '[kwargs]':
                mode = 'kwargs'
            elif layer == '':
                layer = cur_line[1:-1]
            elif mode == 'args':
                try:
                    args.append(eval(cur_line.split('=')[1]))
                except:
                    args.append(cur_line.split('=')[1])
            elif mode == 'kwargs':
                split_line = cur_line.split('=')
                try:
                    kwargs[split_line[0].strip()] = eval(split_line[1].strip())
                except:
                    kwargs[split_line[0].strip()] = split_line[1].strip()
        if layer != '':
            add_layer(model, layer, args,kwargs)
    
    print(model.summary())
    adam = keras.optimizers.Adam()
    #adagrad = Adagrad()
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        print('RAM Usage {:.2f} GB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))
        
        

if __name__ == "__main__":
    args = parseArguments()
    model = base_model(args.model)
    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png')
    #print model.summary()
