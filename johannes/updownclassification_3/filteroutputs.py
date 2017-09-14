
# coding: utf-8

# In[1]:

import jkutils, updown_network
import os, sys, math
import numpy as np
import scipy.stats as stats
#from scipy.misc import imsave
import time
from keras import backend as K
from keras.models import load_model
import math
import shelve
import itertools


# In[2]:



# In[3]:

# ## load project

# In[4]:

#time
today = '2017-09-07'
project_name = 'time_high_v2_lb'


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
projectpath=os.path.join(file_location,project_folder)
train_inds = shelf['Train_Inds'] 
valid_inds = shelf['Valid_Inds']
test_inds = shelf['Test_Inds']
inf_times_as = shelf['inf_times_as']
norm = shelf['time_normalized'] if shelf.has_key('time_normalized') else False
test_results = np.load(os.path.join(projectpath, test_results))
input_data, out_data, file_len = jkutils.read_files(
    input_files, 
    data_location, 
    using=shelf['using'])
res, test_out, zenith_out = test_results[0,:], test_results[1,:], test_results[2,:] #network output (0 or 1), 
                                                                                    #desired output (0 or 1),
                                                                                    #zenith (0 to pi)


# In[6]:


# In[7]:

model=load_model(os.path.join(projectpath, 'final_network.h5'))


# In[7]:


# In[63]:

ml = model.layers[0]


# In[65]:

# # display filter-outputs for real inputs #

# In[9]:

# choose the first n events that are correctly reconstructed
n = 50
event_mask = (np.round(res) == test_out)
c=0
for i, d in enumerate(event_mask):
    if d: 
        c+=1
    if c == n: 
        c = i
        break
event_mask[c+1:]=False


# In[13]:

if n > (test_inds[0][1] - test_inds[0][0]):
    print "choose a smaller n"
    sys.exit()
gen = updown_network.generator(n, input_data, out_data, 
                                        [list(np.arange(*test_inds[0])[event_mask])],
                                        inf_times_as=inf_times_as, normalize=norm)
#event_inputs = event[0]
#event_outputs = event[1]
zeniths = zenith_out[event_mask]
energies = out_data[0][list(np.arange(*test_inds[0])[event_mask]),"energy"]
convlayer = filter(lambda f: 'conv' in f.name, model.layers)
filter_outputs = [{f.name : np.zeros((f.filters)) for f in convlayer} 
                  for i in range(n)]


# In[16]:

# In[79]:


# In[ ]:

absolute = False

for i, event in enumerate(gen.next()[0]):
    img_shape = event.squeeze().shape
    input_img_data = event
    if K.image_data_format() == 'channels_first':
        input_img_data = input_img_data.reshape((1, 1)+img_shape)
    else:
        input_img_data = input_img_data.reshape((1,) + img_shape + (1,))
    orig = input_img_data.copy()
    # we scan through every conv layer and every filter
    print('Processing event {}. Zenith: {:.2f}. Energy: {}'.format(
            i, zeniths[i], energies[i]))
    
    # we build a loss function that calculates the mean activation
    # of the nth filter of the layer considered
    for layer in convlayer:
        print '  ', layer.name, 
        layer_output = layer.output
        for filter_index in range(layer.filters):
            input_img=model.input
            if K.image_data_format() == 'channels_first':
                # (batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)
                act = K.mean(layer_output[:, filter_index, :, :, :])
            else:
                # (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)
                act = K.mean(layer_output[:, :, :, :, filter_index])

            if absolute:
                act = K.abs(loss)

            # this function returns the act value given the input picture
            actval = K.function([input_img, K.learning_phase()], [act])

            val = actval([input_img_data, 1])

            sys.stdout.write('.')
            filter_outputs[i][layer.name][filter_index] = val[0]
        print

#np.save("kept_{}{}_{}".format(layer_name, "_abs" if absolute else "" , "-".join(project_folder.split("/")[1:])),kept_filters)


# In[ ]:

np.save("fouts_{}events{}_{}_{}_script".format(n, "_abs" if absolute else "" , "-".join(project_folder.split("/")[1:])), filter_outputs)


# In[97]:

# In[ ]:



