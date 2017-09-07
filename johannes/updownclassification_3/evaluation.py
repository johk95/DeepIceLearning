#!/usr/bin/env python
# coding: utf-8

# # see what angles get reconstructed well

# In[2]:

import jkutils
import os, sys
import numpy as np
import scipy.stats as stats
from keras.models import load_model
import math
import shelve
import itertools


# # analyze test_results

# In[3]:

#charge
today = '2017-08-24'
project_name = 'charge_h012_v1'


# In[4]:

#time
today = '2017-08-24'
project_name = 'time_h012_v1'


# In[5]:

file_location = '/data/user/jkager/NN_Reco/johannes/updownclassification_3/'
data_location = '/data/user/jkager/NN_Reco/training_data_20x10x60/'
test_results = 'test_results.npy'

project_folder = 'train_hist/{}/{}'.format(today, project_name)
print "looking for", project_folder
if not os.path.exists(os.path.join(file_location,project_folder)):
    print "project not found. exiting..."
    sys.exit(-1)
elif not os.path.exists(os.path.join(file_location, project_folder, test_results)):
    print "test results not found. exiting..."
    sys.exit(-1)
print "found"
shelf = shelve.open(os.path.join(file_location, project_folder, 'run_info.shlf'))
input_files = shelf['Files'].split(':')
if len(input_files) == 1: #this could be something like ['h01'] (inputformat)
    #try to decode fileinput format
    input_files = jkutils.get_filenames(input_files[0])
    for f in input_files:
        if not os.path.isfile(os.path.join(data_location, 'training_data/{}'.format(f))):
            print "file not found:", f
            print "exiting script."
            sys.exit(1)
train_inds = shelf['Train_Inds'] 
valid_inds = shelf['Valid_Inds']
test_inds = shelf['Test_Inds']
test_results = np.load(os.path.join(file_location, project_folder, test_results))
input_data, out_data, file_len = jkutils.read_files(input_files, data_location, using=shelf['using'])


# res is the output value of the network. test_out is the expected value (0 or 1, up or down depending on the real zenith value). zenith_out is the real zenith value

# In[6]:

res, test_out, zenith_out = test_results[0,:], test_results[1,:], test_results[2,:] #network output (0 or 1), 
                                                                                    #desired output (0 or 1),
                                                                                    #zenith (0 to pi)



# ## accuracy over number of hit bins

# In[12]:

def num_hit_bins(input_set):
    ret = 0
    for j in input_set.flatten():
        if shelf['using'] == 'time' and j != np.inf:
            ret += 1
        if shelf['using'] == 'charge' and j > 0.0:
            ret += 1
    return ret


# In[13]:

def get_pos_in_res(i_file, i_in_testsets):
    before = sum([test_inds[i][1] - test_inds[i][0] for i in range(i_file)])
    return before + i_in_testsets


# In[ ]:

bins = 100
input_shape = input_data[0].shape[1:-1]
x_bins = np.linspace(0,reduce(lambda x, y: x*y, input_shape),bins)
y_acc_data = [[] for i in range(bins-1)]
cor, summe = 0, 0
for file_n in range(len(input_data)):
    for i, inp_s in enumerate(input_data[file_n][test_inds[file_n][0]:test_inds[file_n][1]]):
        n_hit = num_hit_bins(inp_s)
        if n_hit == x_bins[-1]:
            bin_n = bins - 2
        else:
            bin_n = np.digitize(np.array([n_hit]), x_bins)[0] - 1
        #remember if it was correctly reconstructed
        index = get_pos_in_res(file_n, i)
        correct = np.round(res[index]) == jkutils.zenith_to_binary(zenith_out[index]) #latter is same as tet_out
        y_acc_data[bin_n].append(correct)

np.save("y_acc_full_{}_{}".format(today.replace('-',''), project_name), y_acc)
y_acc = [float(sum(i))/len(i) if len(i) > 0 else None for i in y_acc_data]
np.save("y_acc_time_h012_v1_240817",y_acc)
y_acc_zero = [i if i is not None else 0.0 for i in y_acc]
np.save("y_acc_zero_h012_v1_240817",y_acc_zero)


