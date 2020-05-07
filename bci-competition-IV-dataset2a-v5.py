#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<style>\n.code_cell .run_this_cell {\n    display: none;\n}\n</style>')


# After meeting on Tuesday 5 May 2020:

# To do:
# * [ ] Check the classifier, try improve the accuracy, how?
#     * [x] Try using np.cov
#     * [x] Try applying the CAR filter
#     * [x] Try remove the artifacts
# * [ ] How to verify training dataset
# * [x] go to bci repository, understand their code
# * [x] Check the data workflow, is it correct?
# * [x] David suggest to also check the train model on train data
# * [ ] figure of PSD of C3 and C4 over average of all trials each subject, one figure has plots of right and left in diff color,  so in total there will be 18 figure 9 C3 and 9 C4, ERD will decrease in beta band, notice what happened in alpha band
# 
# Etc, personal:
# * [ ] The code is too long after finish import all function
# 
# Still looking for the best way to 
# ___

# For improving the classifier, try different combination of steps above
# * Model 1
#     * Apply CAR filter
#     * *FAILED*
# * Model 2
#     * np.cov
#     * Apply CAR filter
#     * *FAILED*
# * Model 3
#     * Apply CAR filter
#     * remove artifact
#     * *FAILED*
# * Model 4
#     * np.cov
#     * Apply CAR filter
#     * remove artifact
#     * *FAILED*
# * Model 5
#     * np.cov
#     * *DONE, still low acc*
# * Model 6
#     * np.cov
#     * remove artifact
#     * *DONE, still low acc*
# * Model 7 
#     * remove artifact
#     * *DONE, still low acc*

# In[1]:


get_ipython().run_line_magic('autosave', '5')


# ---
# This file will use the train_test_split function provided in sklearn_metrics  
# Thus splitting training and testing data will not be performed manually  
# ___
# In addition to that the overlapping data will be performed before CSP calculation
# The amount of overlapping performed on the data are:
# 1. 10%
# 2. 50%
# 3. 90%
# 

# # BCI Competition IV Dataset 2a (.npz data)
# <p>
# Information Given in Documentation
# 
# From the documentation it is known that:
# <li>25 electrodes are used, first 22 are EEG, last 3 are EOG
# <li>Sampling frequency (fs) is 250Hz
# <li>9 subjects
# <li>9 run (run 1-3 are for eye movement, run 4-9 is MI)
# 
# <b> -- Time Duration-- </b>
# 
# 1 trials                          = 7-8s  
# 1 run              = 48 trials    = 336-384s  
# 1 session = 6 runs = 288 trials   = 2016-2304s
# 
# About the recording of eye movement
# <li>run 1 => 2 mins with eyes open
# <li>run 2 => 1 min with eyes closed
# <li>run 3 => 1 min with eye movements

# ## Visualizing/ Preparing the Data 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import pandas as pd
from pandas import DataFrame as dframe


# In[3]:


np.seterr(divide='ignore', invalid='ignore')


# In[4]:


# np.set_printoptions(precision=30, suppress=True)


# In[5]:


# number of subject
ns = 9


# In[6]:


# Load the training data of subject 01
data01 = np.load('./datasets/A01T.npz')
data02 = np.load('./datasets/A02T.npz')
data03 = np.load('./datasets/A03T.npz')
data04 = np.load('./datasets/A04T.npz')
data05 = np.load('./datasets/A05T.npz')
data06 = np.load('./datasets/A06T.npz')
data07 = np.load('./datasets/A07T.npz')
data08 = np.load('./datasets/A08T.npz')
data09 = np.load('./datasets/A09T.npz')


# In[7]:


type(data01)


# In[8]:


# Keys available inside data are
data01.files


# In[9]:


data09['artifacts'].shape # Containing artifacts occurent in each event in all runs of a subject


# In[10]:


dframe(data09['artifacts'])


# In[11]:


# Create table with three columns of 'etyp', 'epos', 'edur' respectively
# Name this table property
prop01 = np.c_[data01['etyp'], data01['epos'], data01['edur']]
prop02 = np.c_[data02['etyp'], data02['epos'], data02['edur']]
prop03 = np.c_[data03['etyp'], data03['epos'], data03['edur']]
prop04 = np.c_[data04['etyp'], data04['epos'], data04['edur']]
prop05 = np.c_[data05['etyp'], data05['epos'], data05['edur']]
prop06 = np.c_[data06['etyp'], data06['epos'], data06['edur']]
prop07 = np.c_[data07['etyp'], data07['epos'], data07['edur']]
prop08 = np.c_[data08['etyp'], data08['epos'], data08['edur']]
prop09 = np.c_[data09['etyp'], data09['epos'], data09['edur']]


# In[12]:


prop01.shape


# In[13]:


pd.__version__


# In[14]:


pd.options.display.max_rows = None


# In[15]:


dframe(prop02, columns=['etype', 'epos','edur'])


# In[16]:


# Store all 's' data into sample_data
sample_data01 = dframe(data01['s'])
sample_data02 = dframe(data02['s'])
sample_data03 = dframe(data03['s'])
sample_data04 = dframe(data04['s'])
sample_data05 = dframe(data05['s'])
sample_data06 = dframe(data06['s'])
sample_data07 = dframe(data07['s'])
sample_data08 = dframe(data08['s'])
sample_data09 = dframe(data09['s'])


# In[17]:


# Remove the last 3 electrodes (EOG electrodes)
sample_data01 = sample_data01.iloc[:, 0:22]
sample_data02 = sample_data02.iloc[:, 0:22]
sample_data03 = sample_data03.iloc[:, 0:22]
sample_data04 = sample_data04.iloc[:, 0:22]
sample_data05 = sample_data05.iloc[:, 0:22]
sample_data06 = sample_data06.iloc[:, 0:22]
sample_data07 = sample_data07.iloc[:, 0:22]
sample_data08 = sample_data08.iloc[:, 0:22]
sample_data09 = sample_data09.iloc[:, 0:22]


# In[21]:


sample_data01.shape


# In[22]:


# Showing inital value of sample_data
sample_data01.head()


# In[23]:


# Sample data has shape of 67k~ samples
# This samples belong to total of all 9 runs with 48 trials of 4 different classes
for i in range(1, 10):
    var = 'sample_data0' + str(i)
    print(var, globals()[var].shape)


# In[ ]:





# ## Remove Artifact
# Inside data of each subject exists a 'artifacts' data which contains the information on each trial whether the data is clean, denoted by `0` or containing artifact, denoted by `1`

# In[ ]:





# In[ ]:





# In[ ]:


# Append the artifacts column to all events data, shape 288 x 4
new_prop = np.hstack([all_ev, data['artifacts']])


# In[ ]:


def removeArtifacts(prop, data):
    # From prop take all data corresponds to event 769, 770, 771, 772
    idx = np.asarray(np.where((prop[:, 0]>768) & (prop[:, 0]<773))).flatten()
    all_ev = prop[idx, :]

    # Append the artifacts column to all events data, shape 288 x 4
    new_prop = np.hstack([all_ev, data['artifacts']])

    # Find index of clean data from all_ev with classes 769 (left) or 770 (right) and artifacts equal to '1'
    idx_cl = np.argwhere(((new_prop[:, 0]==769) | (new_prop[:, 0]==770)) & (new_prop[:, 3]==0)).flatten()
    clean_ev = new_prop[idx_cl, :]

    return clean_ev


# In[ ]:


# Remove artifacts and take only event 769 (left) and 770 (right)
clean_ev01 = removeArtifacts(prop01, data01)
clean_ev02 = removeArtifacts(prop02, data02)
clean_ev03 = removeArtifacts(prop03, data03)
clean_ev04 = removeArtifacts(prop04, data04)
clean_ev05 = removeArtifacts(prop05, data05)
clean_ev06 = removeArtifacts(prop06, data06)
clean_ev07 = removeArtifacts(prop07, data07)
clean_ev08 = removeArtifacts(prop08, data08)
clean_ev09 = removeArtifacts(prop09, data09)


# In[ ]:


dframe(clean_ev07)


# In[ ]:


# Fetch indexes whose 1st column are 769 (left) and 770 (right)
# Subject 1 - 9
for i in range(1, 10):
    idx_l = 'idx0' + str(i) + '_l'
    idx_r = 'idx0' + str(i) + '_r'
    clean_ev = globals()['clean_ev0' + str(i)]
    globals()[idx_l] = np.argwhere(clean_ev[:, 0]==769).flatten()
    globals()[idx_r] = np.argwhere(clean_ev[:, 0]==770).flatten()


# In[ ]:


idx01_r


# In[ ]:


# Fetch positions of corresponding event types
# Subject 01-09
for i in range(1, 10):
    pos_l = 'pos0' + str(i) + '_l'
    pos_r = 'pos0' + str(i) + '_r'
    clean_ev = globals()['clean_ev0' + str(i)]
    idx_l = globals()['idx0' + str(i) + '_l']
    idx_r = globals()['idx0' + str(i) + '_r']

    globals()[pos_l] = clean_ev[idx_l, 1]
    globals()[pos_r] = clean_ev[idx_r, 1]


# In[ ]:


pos07_r


# In[ ]:


# Duration of both events lasted for 313 samples, we can defined dur as
cue_dur = 313


# In[ ]:


idx02_l.shape


# In[ ]:





# ## Without artifacts removal

# According to the documentation we are interested to grab event type of 769, left class and 770, right class
# 
# With each type has duration of 313 samples

# In[24]:


# Fetch indexes whose 1st column are 769 (left) and 770 (right)
# Subject 1 - 9
for i in range(1, 10):
    idx_l = 'idx0' + str(i) + '_l'
    idx_r = 'idx0' + str(i) + '_r'
    prop = globals()['prop0' + str(i)]
    globals()[idx_l] = np.argwhere(prop[:, 0]==769).flatten()
    globals()[idx_r] = np.argwhere(prop[:, 0]==770).flatten()


# In[25]:


idx09_l.shape, idx09_r.shape


# In[26]:


dframe(idx05_l).tail()


# In[27]:


# Fetch positions of corresponding event types
# Now this array contains index on which an event begins
# Subject 01-09
for i in range(1, 10):
    pos_l = 'pos0' + str(i) + '_l'
    pos_r = 'pos0' + str(i) + '_r'
    prop = globals()['prop0' + str(i)]
    idx_l = globals()['idx0' + str(i) + '_l']
    idx_r = globals()['idx0' + str(i) + '_r']

    globals()[pos_l] = prop[idx_l, 1]
    globals()[pos_r] = prop[idx_r, 1]


# In[28]:


pos09_l.shape, pos09_r.shape


# In[29]:


dframe(pos04_r)


# In[30]:


# Grab the duration of each event
# Subject 01-09
for i in range(1, 10):
    dur_l = 'dur0' + str(i) + '_l'
    dur_r = 'dur0' + str(i) + '_r'
    prop = globals()['prop0' + str(i)]
    idx_l = globals()['idx0' + str(i) + '_l']
    idx_r = globals()['idx0' + str(i) + '_r']
    
    globals()[dur_l] = prop[idx_l, 2]
    globals()[dur_r] = prop[idx_r, 2]    


# In[31]:


dur01_l


# In[32]:


# Duration of both events lasted for 313 samples, we can defined dur as
cue_dur = 313


# Be aware that this duration is actually the duration of the cue, let's take a look at the event type table in datasheet
# ![event-type-table.png](./img/event-type-table.png)
# 
# The timing scheme of the paradigm suggest that the MI task last for about 2.75 since after the cue ends
# ![timing-scheme-paradigm.png](./img/timing-scheme-paradigm.png)

# In[33]:


# The amount of sample to be clipped for 1 trial of left and right
sec = 2.75
fs = 250
mi_dur = round(sec * fs)
print('After the 313 samples cue, clip this amount of samples:', mi_dur)


# ## Fetch 688 samples of each event from sample_data, 72 trials each class

# In[34]:


len(pos02_l)


# In[35]:


# Fetch 688 samples of each event from sample_data
# Store them in E_left and E_right respectively, For convenience 'E' is used to imitate the variable used in paper
# Preparing the variables to store event samples of shape 72 x 688 x 22
# Subject 01-09
for i in range(1, ns+1):
    E_l = 'E0' + str(i) + '_left'
    E_r = 'E0' + str(i) + '_right'
    pos_l = globals()['pos0' + str(i) + '_l']
    pos_r = globals()['pos0' + str(i) + '_r']
    
    sample = globals()['sample_data0' + str(i)]
    
    globals()[E_l] = np.zeros([pos_l.shape[0], mi_dur, sample.shape[1]])
    globals()[E_r] = np.zeros([pos_r.shape[0], mi_dur, sample.shape[1]])


# In[36]:


E04_left.shape, E04_right.shape


# In[37]:


pos04_r


# In[38]:


dframe(prop05)


# In[39]:


# Now we are ready to take 688 samples of left and right
# Subject 01-09
for i in range(1, 10):
    E_l = globals()['E0' + str(i) + '_left']
    E_r = globals()['E0' + str(i) + '_right']
    
    sample = globals()['sample_data0' + str(i)]

    pos_l = globals()['pos0' + str(i) + '_l']
    pos_r = globals()['pos0' + str(i) + '_r']
    
    for j in range(E_l.shape[0]):
        E_l[j]=sample.iloc[pos_l[j]+cue_dur:pos_l[j]+cue_dur+mi_dur, :]
    for k in range(E_r.shape[0]):
        E_r[k]=sample.iloc[pos_r[k]+cue_dur:pos_r[k]+cue_dur+mi_dur, :]


# In[40]:


# Creating columns for sample data
Xcol = []
for i in range(1, 23):
    if i < 11:
        Xcol.append('EEG0'+str(i))
    else:
        Xcol.append('EEG'+str(i))


# In[41]:


E01_left.shape


# In[42]:


dframe(E01_left[60], columns=Xcol)


# In[43]:


# First transform the dimension of array to N x T, i.e. electrodes x  samples
# Subject 01-09
for i in range(1, ns+1):
    E_l = globals()['E0' + str(i) + '_left']
    E_r = globals()['E0' + str(i) + '_right']
    
#     try this:
    E_l_var = 'E0' + str(i) + '_left'
    E_r_var = 'E0' + str(i) + '_right'

    
    if (E_l.shape[0] != 22 and E_r.shape[0] !=22 ):
        globals()[E_l_var] = np.transpose(E_l, (0, 2, 1))
        globals()[E_r_var] = np.transpose(E_r, (0, 2, 1))


# In[44]:


E02_left.shape, E02_right.shape


# In[45]:


dframe(E09_left[50]).head()


# In[ ]:





# ## Try With CAR

# In[ ]:


for i in range(1, ns+1):
    CAR_l = 'CAR0' + str(i) + '_left'
    CAR_r = 'CAR0' + str(i) + '_right'
    
    E_l = globals()['E0' + str(i) + '_left']
    E_r = globals()['E0' + str(i) + '_right']
    
    globals()[CAR_l], globals()[CAR_r] = CAR(E_l, E_r)


# In[ ]:


CAR05_left.shape, CAR05_right.shape


# In[ ]:


dframe(CAR03_left[0]).head()


# ## Perform Overlap on Data

# ![overlap-eq.png](./img/overlap_equation.png)

# In[46]:


# 

def overlap(fs, E_data, overlap):
    '''
    This function will return array of overlapped data in sample domain
    Parameter: 
    fs : sampling frequency
    E_data : original EEG sample data, dimension of trial x electrode x samples
    overlap : % data to be overlapped
    '''
    old, nel, dur = E_data.shape # no. of old trials 72, electrode 22, samples 688
    inc = int((round(1-overlap,1))*fs) # increment value
    n   = int((dur - fs)/inc) + 1 # no. of portion single trial is sliced into
    new = n * old # no. of new trial
    
    # Prepare the array to store overlapped data
    # Should be dimension of no.trials*fold x no. of eletrode x np. of overlapped samples
    E_data_new = np.zeros((old*n, nel, fs))
    
    # Overlapping data
    for i in range(old):
        for j in range(n):
            temp = E_data[i][:, j*inc:j*inc+fs]
            E_data_new[(i*n)+(j)] = temp
    
    return E_data_new


# In[47]:


help(overlap)


# In[48]:


# Overlap for 90%
for i in range(1, ns+1):
    E_l_ol90 = 'E0' + str(i) + '_l_ol90'
    E_r_ol90 = 'E0' + str(i) + '_r_ol90'
    
    E_l = globals()['E0' + str(i) + '_left']
    E_r = globals()['E0' + str(i) + '_right']
    
#     # Using CAR
#     E_l = globals()['CAR0' + str(i) + '_left']
#     E_r = globals()['CAR0' + str(i) + '_right']
    
    globals()[E_l_ol90] = overlap(250, E_l, 0.9)
    globals()[E_r_ol90] = overlap(250, E_r, 0.9)
    
# Overlap for 50%
for i in range(1, ns+1):
    E_l_ol50 = 'E0' + str(i) + '_l_ol50'
    E_r_ol50 = 'E0' + str(i) + '_r_ol50'
    
    E_l = globals()['E0' + str(i) + '_left']
    E_r = globals()['E0' + str(i) + '_right']
    
#     # Using CAR
#     E_l = globals()['CAR0' + str(i) + '_left']
#     E_r = globals()['CAR0' + str(i) + '_right']
    
    globals()[E_l_ol50] = overlap(250, E_l, 0.5)
    globals()[E_r_ol50] = overlap(250, E_r, 0.5)
    
# Overlap for 10%
for i in range(1, ns+1):
    E_l_ol10 = 'E0' + str(i) + '_l_ol10'
    E_r_ol10 = 'E0' + str(i) + '_r_ol10'
    
    E_l = globals()['E0' + str(i) + '_left']
    E_r = globals()['E0' + str(i) + '_right']
    
#     # Using CAR
#     E_l = globals()['CAR0' + str(i) + '_left']
#     E_r = globals()['CAR0' + str(i) + '_right']
    
    globals()[E_l_ol10] = overlap(250, E_l, 0.1)
    globals()[E_r_ol10] = overlap(250, E_r, 0.1)


# In[49]:


E01_l_ol90.shape, E01_r_ol90.shape


# In[50]:


E01_l_ol50.shape, E01_r_ol50.shape


# In[51]:


E01_l_ol10.shape, E01_r_ol10.shape


# In[52]:


# The 250 samples data has to be clipped from the normal data in order to match the samples from overlap
for i in range(1,ns+1):
    E_left = globals()['E0' + str(i) + '_left']
    E_right = globals()['E0' + str(i) + '_right']
    
#     # Using CAR    
#     E_left = globals()['CAR0' + str(i) + '_left']
#     E_right = globals()['CAR0' + str(i) + '_right']
    
    E_l = 'E0' + str(i) + '_l'
    E_r = 'E0' + str(i) + '_r'
    
    globals()[E_l] = E_left[:, :, :fs]
    globals()[E_r] = E_right[:, :, :fs]


# In[53]:


E01_l.shape, E01_r.shape


# In[54]:


pd.options.display.max_columns = None


# In[55]:


dframe(E01_l[0]).head()


# In[56]:


dframe(E03_r_ol10[1]).head()


# ## Split train, test, then covariance

# In[57]:


def split3D(E_left, E_right, percent_tr):
    '''
    split3D will received 3D array of left and right trial with dimension of trial x electrodes x samples
    and then split them based on the percentage n_tr
    Parameter: 
        * E_left, E_right, 3D array
        * n_tr, % portion of data to be split as train data, the rest will serve as test data, allowed range(0.0 - 1.0)

    Return: E_left_tr, E_right_tr, E_left_te, E_right_te
    '''
    # Sometimes the left and right class dont have equal clean trial
    if E_left.shape[0] != E_right.shape[0]:
        if E_left.shape[0] < E_right.shape[0]:
            n = E_left.shape[0]
        else:
            n = E_right.shape[0]
    else:
        n = E_left.shape[0]
    
    ntr = round(n*percent_tr)
    nte = n - ntr
    E_left_tr = E_left[:ntr]
    E_right_tr = E_right[:ntr]
    E_left_te = E_left[ntr:ntr+nte]
    E_right_te = E_right[ntr:ntr+nte]
        
    return E_left_tr, E_right_tr, E_left_te, E_right_te
    


# In[58]:


help(split3D)


# In[59]:


# Splitting all data into training and test
# -------- Normal Data ------------
for i in range(1, ns+1):
    E_l_tr = 'E0' + str(i) + '_l_tr'
    E_r_tr = 'E0' + str(i) + '_r_tr'
    
    E_l_te = 'E0' + str(i) + '_l_te'
    E_r_te = 'E0' + str(i) + '_r_te'
    
    E_l = globals()['E0' + str(i) + '_l']
    E_r = globals()['E0' + str(i) + '_r']
    
    globals()[E_l_tr], globals()[E_r_tr],    globals()[E_l_te], globals()[E_r_te] = split3D(E_l, E_r, 0.8)

# -------- 90% Overlapped Data ---------
for i in range(1, ns+1):
    E_l_ol90_tr = 'E0' + str(i) + '_l_ol90_tr'
    E_r_ol90_tr = 'E0' + str(i) + '_r_ol90_tr'
    
    E_l_ol90_te = 'E0' + str(i) + '_l_ol90_te'
    E_r_ol90_te = 'E0' + str(i) + '_r_ol90_te'
    
    E_l_ol90 = globals()['E0' + str(i) + '_l_ol90']
    E_r_ol90 = globals()['E0' + str(i) + '_r_ol90']
    
    globals()[E_l_ol90_tr], globals()[E_r_ol90_tr],    globals()[E_l_ol90_te], globals()[E_r_ol90_te] = split3D(E_l_ol90, E_r_ol90, 0.8)


# -------- 50% Overlapped Data ---------
for i in range(1, ns+1):
    E_l_ol50_tr = 'E0' + str(i) + '_l_ol50_tr'
    E_r_ol50_tr = 'E0' + str(i) + '_r_ol50_tr'
    
    E_l_ol50_te = 'E0' + str(i) + '_l_ol50_te'
    E_r_ol50_te = 'E0' + str(i) + '_r_ol50_te'
    
    E_l_ol50 = globals()['E0' + str(i) + '_l_ol50']
    E_r_ol50 = globals()['E0' + str(i) + '_r_ol50']
    
    globals()[E_l_ol50_tr], globals()[E_r_ol50_tr],    globals()[E_l_ol50_te], globals()[E_r_ol50_te] = split3D(E_l_ol50, E_r_ol50, 0.8)
    
    
# -------- 10% Overlapped Data ---------
for i in range(1, ns+1):
    E_l_ol10_tr = 'E0' + str(i) + '_l_ol10_tr'
    E_r_ol10_tr = 'E0' + str(i) + '_r_ol10_tr'
    
    E_l_ol10_te = 'E0' + str(i) + '_l_ol10_te'
    E_r_ol10_te = 'E0' + str(i) + '_r_ol10_te'
    
    E_l_ol10 = globals()['E0' + str(i) + '_l_ol10']
    E_r_ol10 = globals()['E0' + str(i) + '_r_ol10']
    
    globals()[E_l_ol10_tr], globals()[E_r_ol10_tr],    globals()[E_l_ol10_te], globals()[E_r_ol10_te] = split3D(E_l_ol10, E_r_ol10, 0.8)


# In[60]:


E01_l_ol90_tr.shape, E01_r_ol90_tr.shape, E01_l_ol90_te.shape, E01_r_ol90_te.shape


# In[61]:


E01_l_ol50_tr.shape, E01_r_ol50_tr.shape, E01_l_ol50_te.shape, E01_r_ol50_te.shape


# In[62]:


E01_l_ol10_tr.shape, E01_r_ol10_tr.shape, E01_l_ol10_te.shape, E01_r_ol10_te.shape


# In[63]:


E01_l_tr.shape, E01_r_tr.shape, E01_l_te.shape, E01_r_te.shape


# In[64]:


dframe(E01_r_tr[0]).head()


# ## Processing Training Data

# ### Spatial Covariance and Composite Covariance
# Calculating normalized spatial covariance of each trial and composite variance
# $$ C = \frac{EE'}{trace(EE')}$$
# ' denotes transpose
# 
# Then calculate the average of left and right covariance:
# $$\overline{C}_d, \epsilon[l, r]$$
# 
# The composite average covariance is given as:
# $$ C_c = \overline{C_l} + \overline{C_r}$$
# 
#   
#   

# In[65]:


def compCov(E_left, E_right):
    '''
    compCov receive left and right class 3D array with dimension of trial x electrodes x samples
    Then calculate the covariance of each class, average them, and add them together
    Finally, this function returns 2D composite covariance
    Parameter: 
        * E_left, E_right, each dimension must be the same
    Return: composite averaged covariance
    '''
    nel = E_left.shape[1]
    Cov_left = np.zeros((E_left.shape[0], nel, nel))
    Cov_right = np.zeros((E_right.shape[0], nel, nel))

    for i in range(E_left.shape[0]):
        Cov_left[i, :, :] = (E_left[i, :, :]@E_left[i, :, :].T)/np.trace(E_left[i, :, :]@E_left[i, :, :].T)

    for i in range(E_right.shape[0]):
        Cov_right[i, :, :] = (E_right[i, :, :]@E_right[i, :, :].T)/np.trace(E_right[i, :, :]@E_right[i, :, :].T)
        
    # Average covariance left and right    
    avgCov_l = np.sum(Cov_left, axis=0)/Cov_left.shape[0]
    avgCov_r = np.sum(Cov_right, axis=0)/Cov_right.shape[0]
    
    avgCov_c = avgCov_l + avgCov_r
    return avgCov_c, avgCov_l, avgCov_r
    


# In[66]:


def compCov2(E_left, E_right):
    nel = E_left.shape[1]
    cov_l = np.zeros((E_left.shape[0], nel, nel))
    cov_r = np.zeros((E_right.shape[0], nel, nel))
    
    for i in range(E_left.shape[0]):
        cov_l[i] = np.cov(E_left[i])
    
    for j in range(E_right.shape[0]):
        cov_r[i] = np.cov(E_right[i])
        
    # Average covariance left and right    
    avgCov_l = np.sum(cov_l, axis=0)/cov_l.shape[0]
    avgCov_r = np.sum(cov_r, axis=0)/cov_r.shape[0]
    
    avgCov_c = avgCov_l + avgCov_r
    
    return avgCov_c, avgCov_l, avgCov_r


# In[ ]:


help(compCov2)


# In[67]:


E01_l_ol90_tr.shape, E01_r_ol90_tr.shape, E01_l_ol50_tr.shape, E01_r_ol50_tr.shape


# In[68]:


E01_l_ol10_tr.shape, E01_r_ol10_tr.shape, E01_l_tr.shape, E01_r_tr.shape


# In[69]:


# ============ Normal Data ====================
for i in range(1, ns+1):
    Cov_l_tr = 'Cov0' + str(i) + '_l_tr'
    Cov_r_tr = 'Cov0' + str(i) + '_r_tr'
    Cov_c_tr = 'Cov0' + str(i) + '_c_tr'
    
    E_l_tr = globals()['E0' + str(i) + '_l_tr']
    E_r_tr = globals()['E0' + str(i) + '_r_tr']
    
    globals()[Cov_c_tr], globals()[Cov_l_tr], globals()[Cov_r_tr] = compCov(E_l_tr, E_r_tr)

# ============ 90% Overlap ====================
for i in range(1, ns+1):
    Cov_l_ol90_tr = 'Cov0' + str(i) + '_l_ol90_tr'
    Cov_r_ol90_tr = 'Cov0' + str(i) + '_r_ol90_tr'
    Cov_c_ol90_tr = 'Cov0' + str(i) + '_c_ol90_tr'
    
    E_l_ol90_tr = globals()['E0' + str(i) + '_l_ol90_tr']
    E_r_ol90_tr = globals()['E0' + str(i) + '_r_ol90_tr']
    
    globals()[Cov_c_ol90_tr], globals()[Cov_l_ol90_tr], globals()[Cov_r_ol90_tr] = compCov(E_l_ol90_tr, E_r_ol90_tr)

# ============ 50% Overlap ====================
for i in range(1, ns+1):
    Cov_l_ol50_tr = 'Cov0' + str(i) + '_l_ol50_tr'
    Cov_r_ol50_tr = 'Cov0' + str(i) + '_r_ol50_tr'
    Cov_c_ol50_tr = 'Cov0' + str(i) + '_c_ol50_tr'
    
    E_l_ol50_tr = globals()['E0' + str(i) + '_l_ol50_tr']
    E_r_ol50_tr = globals()['E0' + str(i) + '_r_ol50_tr']
    
    globals()[Cov_c_ol50_tr], globals()[Cov_l_ol50_tr], globals()[Cov_r_ol50_tr] = compCov(E_l_ol50_tr, E_r_ol50_tr)
    
# ============ 10% Overlap ====================
for i in range(1, ns+1):
    Cov_l_ol10_tr = 'Cov0' + str(i) + '_l_ol10_tr'
    Cov_r_ol10_tr = 'Cov0' + str(i) + '_r_ol10_tr'
    Cov_c_ol10_tr = 'Cov0' + str(i) + '_c_ol10_tr'
    
    E_l_ol10_tr = globals()['E0' + str(i) + '_l_ol10_tr']
    E_r_ol10_tr = globals()['E0' + str(i) + '_r_ol10_tr']
    
    globals()[Cov_c_ol10_tr], globals()[Cov_l_ol10_tr], globals()[Cov_r_ol10_tr] = compCov(E_l_ol10_tr, E_r_ol10_tr)


# In[70]:


Cov01_c_ol10_tr.shape


# In[71]:


dframe(Cov03_r_tr).head()


# In[72]:


dframe(Cov02_l_tr).head()


# ### Decompose Covariance Matrix
# Eigendecompose matrix $C_c$, such that
# $$C_c = U_c\lambda_cU'_c$$
# The eigenvalues in this calculated are sorted in descending order

# In[73]:


def decomposeCov(avgCov):
    '''
    This function will decompose covariance matrix of each subject into eigenvec, eigenval, and eigenvec_transpose
    Parameter:
    avgVoc
    
    Return:
    λ_dsc and V_dsc, i.e. eigenvalues and eigenvector in descending order
    
    '''
    λ, V = np.linalg.eig(avgCov)
    λ_dsc = np.sort(λ)[::-1] # Sort eigenvalue descending order, default is ascending order sort
    idx_dsc = np.argsort(λ)[::-1]
    V_dsc = V[:, idx_dsc] # Sort eigenvectors descending order
    λ_dsc = np.diag(λ_dsc) # Diagonalize λ_dsc
    return λ_dsc, V_dsc


# In[74]:


# ------- Normal Data -------------------
for i in range(1, ns+1):
    λ_dsc = 'λ0' + str(i) + '_dsc'
    V_dsc = 'V0' + str(i) + '_dsc'
    
    Cov_tr = globals()['Cov0' + str(i) + '_c_tr']
    
    globals()[λ_dsc], globals()[V_dsc] = decomposeCov(Cov_tr)

# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    λ_ol90_dsc = 'λ0' + str(i) + '_ol90_dsc'
    V_ol90_dsc = 'V0' + str(i) + '_ol90_dsc'
    
    Cov_ol90_tr = globals()['Cov0' + str(i) + '_c_ol90_tr']
    
    globals()[λ_ol90_dsc], globals()[V_ol90_dsc] = decomposeCov(Cov_ol90_tr)
    
# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    λ_ol50_dsc = 'λ0' + str(i) + '_ol50_dsc'
    V_ol50_dsc = 'V0' + str(i) + '_ol50_dsc'
    
    Cov_ol50_tr = globals()['Cov0' + str(i) + '_c_ol50_tr']
    
    globals()[λ_ol50_dsc], globals()[V_ol50_dsc] = decomposeCov(Cov_ol50_tr)
    
# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    λ_ol10_dsc = 'λ0' + str(i) + '_ol10_dsc'
    V_ol10_dsc = 'V0' + str(i) + '_ol10_dsc'
    
    Cov_ol10_tr = globals()['Cov0' + str(i) + '_c_ol10_tr']
    
    globals()[λ_ol10_dsc], globals()[V_ol10_dsc] = decomposeCov(Cov_ol10_tr)


# In[75]:


dframe(λ04_dsc)


# ### White Transformation Matrix P 
# Construct the whiten transformation matrix P:
# $$P = \sqrt{\lambda_c^{-1}}U'_c$$
# <br>
# The whitening transformation equalizes the variances in the space spanned by $U_c$, i.e., all eigenvalues of ${PC_CP'}$ are equal to identity matrix.

# In[76]:


def whiteMatrix(λ_dsc, V_dsc):
    '''
    '''
    λ_dsc_sqr = np.sqrt(np.linalg.inv(λ_dsc))
    P = (λ_dsc_sqr)@(V_dsc.T)
    
    return P

def ifWhiteIdentity(P, Cov_c):
    '''
    '''
    # Try if eigenval of PCcP' equal to identity matrix
    temp = P@Cov_c@P.T
    temp = np.diag(np.linalg.eigvals(temp))
    
    return abs(temp)


# In[77]:


# Computing White Transformation Matrix P
print('Computing white transformation matrix....\n')

# ------- Normal Data -------------------
for i in range(1, ns+1):
    λ_dsc = globals()['λ0' + str(i) + '_dsc']
    V_dsc = globals()['V0' + str(i) + '_dsc']
    P = 'P0' + str(i)
    
    globals()[P] = whiteMatrix(λ_dsc, V_dsc)


# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    λ_ol90_dsc = globals()['λ0' + str(i) + '_ol90_dsc']
    V_ol90_dsc = globals()['V0' + str(i) + '_ol90_dsc']
    
    P_ol90 = 'P0' + str(i) + '_ol90'
    
    globals()[P_ol90] = whiteMatrix(λ_ol90_dsc, V_ol90_dsc)

# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    λ_ol50_dsc = globals()['λ0' + str(i) + '_ol50_dsc']
    V_ol50_dsc = globals()['V0' + str(i) + '_ol50_dsc']
    
    P_ol50 = 'P0' + str(i) + '_ol50'
    
    globals()[P_ol50] = whiteMatrix(λ_ol50_dsc, V_ol50_dsc)

# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    λ_ol10_dsc = globals()['λ0' + str(i) + '_ol10_dsc']
    V_ol10_dsc = globals()['V0' + str(i) + '_ol10_dsc']
    
    P_ol10 = 'P0' + str(i) + '_ol10'
    
    globals()[P_ol10] = whiteMatrix(λ_ol10_dsc, V_ol10_dsc)

print('White transfomartion has dimension of: ', P01.shape)


# In[78]:


P01_ol90.shape


# In[79]:


dframe(P05_ol50).tail()


# In[80]:


# Try if eigenval of PCcP' equal to identity matrix
temp = ifWhiteIdentity(P05, Cov05_c_tr)


# In[81]:


dframe(temp)


# ### Spatial Matrix
# Spatial matrix is a transformation of $\overline{C_l}$ and $\overline{C_r}$ into:
# $$S_1 = P\overline{C_1}P'$$  and   $$S_r = P\overline{C_r}P'$$

# In[82]:


def spatial(P, Cov_l, Cov_r):
    # Use this whitening matrix transform Sl and Sr as follows
    # Sl = P Cl P'
   
    S_left = P@Cov_l@P.T
    # Sr = P Cr P'
    S_right = P@Cov_r@P.T
    return S_left, S_right


# In[83]:


Cov01_l_tr.shape


# In[84]:


# ------- Normal Data -------------------
for i in range(1, ns+1):
    S_l = 'S0' + str(i) + '_l'
    S_r = 'S0' + str(i) + '_r'
    
    P = globals()['P0' + str(i)]
    
    Cov_l = globals()['Cov0' + str(i) + '_l_tr']
    Cov_r = globals()['Cov0' + str(i) + '_r_tr']
    
    globals()[S_l], globals()[S_r] = spatial(P, Cov_l, Cov_r)

    
# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    S_l_ol90 = 'S0' + str(i) + '_l_ol90'
    S_r_ol90 = 'S0' + str(i) + '_r_ol90'
    
    P_ol90 = globals()['P0' + str(i) + '_ol90']
    
    Cov_l_ol90_tr = globals()['Cov0' + str(i) + '_l_ol90_tr']
    
    Cov_r_ol90_tr = globals()['Cov0' + str(i) + '_r_ol90_tr']

    
    globals()[S_l_ol90], globals()[S_r_ol90] = spatial(P_ol90, Cov_l_ol90_tr, Cov_r_ol90_tr)

# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    S_l_ol50 = 'S0' + str(i) + '_l_ol50'
    S_r_ol50 = 'S0' + str(i) + '_r_ol50'
    
    P_ol50 = globals()['P0' + str(i) + '_ol50']
    
    Cov_l_ol50_tr = globals()['Cov0' + str(i) + '_l_ol50_tr']
    
    Cov_r_ol50_tr = globals()['Cov0' + str(i) + '_r_ol50_tr']

    
    globals()[S_l_ol50], globals()[S_r_ol50] = spatial(P_ol50, Cov_l_ol50_tr, Cov_r_ol50_tr)

# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    S_l_ol10 = 'S0' + str(i) + '_l_ol10'
    S_r_ol10 = 'S0' + str(i) + '_r_ol10'
    
    P_ol10 = globals()['P0' + str(i) + '_ol10']
    
    Cov_l_ol10_tr = globals()['Cov0' + str(i) + '_l_ol10_tr']
    
    Cov_r_ol10_tr = globals()['Cov0' + str(i) + '_r_ol10_tr']

    
    globals()[S_l_ol10], globals()[S_r_ol10] = spatial(P_ol10, Cov_l_ol10_tr, Cov_r_ol10_tr)
    


# In[85]:


S02_l.shape, S02_r.shape, S02_l_ol10.shape, S02_r_ol10.shape


# In[86]:


dframe(S05_r_ol90).head()


# ### Projection matrix W

# #### 1. Decomposing $Sl$ and $Sr$
# 
# Decompose $S_l$ and $S_r$ such that:
# $$S_l = B{\lambda_l}B'$$ and 
# $$S_r = B{\lambda_r}B'$$
# <br>
# Both $S_l$ and $S_r$ share common eigenvalues but in opposite magnitude order, thus by ordering them in ascending and descending order respectively, we can get<br> 
# $$\lambda_l + \lambda_r = I$$  

# In[87]:


'''
'''
def decomposeSpatial(S_left, S_right):
    # Decompose Sl and Sr
    λ_left, B_left = np.linalg.eig(S_left)
    λ_right, B_right = np.linalg.eig(S_right)
    
    # Sort left eigenvalues, ascending order
    idx_asc = λ_left.argsort() # Use this index to sort eigenvector smallest -> largest
    λ_left_asc = λ_left[idx_asc]
    
    # Sort left eigenvector, ascending order
    B_left = B_left[:, idx_asc]
    
    # Sort right eigenvector, descending order
    idx_dsc = λ_right.argsort()[::-1] # Use this index to sort eigenvector largest -> smallest
    λ_right_dsc = λ_right[idx_dsc]
    
    # Sort right eigenvector, descending order
    B_right = B_right[:, idx_dsc]
    
    return B_left, B_right, λ_left_asc, λ_right_dsc 

'''
'''
def ifSpatialIdentity(λ_left_asc, λ_right_dsc):
    ones = λ_left_asc + λ_right_dsc
    return ones


# In[88]:


# ------- Normal Data -------------------
for i in range(1, ns+1):
    S_l = globals()['S0' + str(i) + '_l']
    S_r = globals()['S0' + str(i) + '_r']
    
    B_l = 'B0' + str(i) + '_l'
    B_r = 'B0' + str(i) + '_r'
    
    λ_l_asc = 'λ0' + str(i) + '_l_asc'
    λ_r_dsc = 'λ0' + str(i) + '_r_dsc'
    
    globals()[B_l], globals()[B_r],    globals()[λ_l_asc],  globals()[λ_r_dsc] = decomposeSpatial(S_l, S_r)


# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    S_l_ol90 = globals()['S0' + str(i) + '_l_ol90']
    S_r_ol90 = globals()['S0' + str(i) + '_r_ol90']
    
    B_l_ol90 = 'B0' + str(i) + '_l_ol90'
    B_r_ol90 = 'B0' + str(i) + '_r_ol90'
    
    λ_l_asc_ol90 = 'λ0' + str(i) + '_l_asc_ol90'
    λ_r_dsc_ol90 = 'λ0' + str(i) + '_r_dsc_ol90'
    
    globals()[B_l_ol90], globals()[B_r_ol90],    globals()[λ_l_asc_ol90], globals()[λ_r_dsc_ol90] = decomposeSpatial(S_l_ol90, S_r_ol90)

    
# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    S_l_ol50 = globals()['S0' + str(i) + '_l_ol50']
    S_r_ol50 = globals()['S0' + str(i) + '_r_ol50']
    
    B_l_ol50 = 'B0' + str(i) + '_l_ol50'
    B_r_ol50 = 'B0' + str(i) + '_r_ol50'
    
    λ_l_asc_ol50 = 'λ0' + str(i) + '_l_asc_ol50'
    λ_r_dsc_ol50 = 'λ0' + str(i) + '_r_dsc_ol50'
    
    globals()[B_l_ol50], globals()[B_r_ol50],    globals()[λ_l_asc_ol50], globals()[λ_r_dsc_ol50] = decomposeSpatial(S_l_ol50, S_r_ol50)

    
# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    S_l_ol10 = globals()['S0' + str(i) + '_l_ol10']
    S_r_ol10 = globals()['S0' + str(i) + '_r_ol10']
    
    B_l_ol10 = 'B0' + str(i) + '_l_ol10'
    B_r_ol10 = 'B0' + str(i) + '_r_ol10'
    
    λ_l_asc_ol10 = 'λ0' + str(i) + '_l_asc_ol10'
    λ_r_dsc_ol10 = 'λ0' + str(i) + '_r_dsc_ol10'
    
    globals()[B_l_ol10], globals()[B_r_ol10],    globals()[λ_l_asc_ol10], globals()[λ_r_dsc_ol10] = decomposeSpatial(S_l_ol10, S_r_ol10)


# In[89]:


λ09_r_dsc_ol90 + λ09_l_asc_ol90


# In[90]:


ifSpatialIdentity(λ08_l_asc_ol50, λ08_r_dsc_ol50)


# In[91]:


B01_l.shape


# In[93]:


dframe(B03_r).head()


# In[92]:


dframe(B03_l).head()


# #### 2. Computing Projection Matrix
# Projection Matrix W is given as:
# $$W = (B'P)'$$
# 
# ' denotes transposed

# In[94]:


def projectionMatrix(B_left, B_right, P):
    # Since Bl and Br share common values, we can pick one of them as B
    # To verify try compute W1 and W2 using Br and Bl respectively
    W1 = (B_right.T@P).T
    W2 = (B_left.T@P).T
    return W1, W2


# In[95]:


# ------- Normal Data -------------------
for i in range(1, ns+1):
    B_l = globals()['B0' + str(i) + '_l']
    B_r = globals()['B0' + str(i) + '_r']
    
    P = globals()['P0' + str(i)]
    
    W1 = 'W0' + str(i) + '_1'
    W2 = 'W0' + str(i) + '_2'
    
    globals()[W1], globals()[W2] = projectionMatrix(B_l, B_r, P)
    
# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    B_l_ol90 = globals()['B0' + str(i) + '_l_ol90']
    B_r_ol90 = globals()['B0' + str(i) + '_r_ol90']
    
    P_ol90 = globals()['P0' + str(i) + '_ol90']
    
    W_1_ol90 = 'W0' + str(i) + '_1_ol90'
    W_2_ol90 = 'W0' + str(i) + '_2_ol90'
    
    globals()[W_1_ol90], globals()[W_2_ol90] = projectionMatrix(B_l_ol90, B_r_ol90, P_ol90)
    
# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    B_l_ol50 = globals()['B0' + str(i) + '_l_ol50']
    B_r_ol50 = globals()['B0' + str(i) + '_r_ol50']
    
    P_ol50 = globals()['P0' + str(i) + '_ol50']
    
    W_1_ol50 = 'W0' + str(i) + '_1_ol50'
    W_2_ol50 = 'W0' + str(i) + '_2_ol50'
    
    globals()[W_1_ol50], globals()[W_2_ol50] = projectionMatrix(B_l_ol50, B_r_ol50, P_ol50)

# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    B_l_ol10 = globals()['B0' + str(i) + '_l_ol10']
    B_r_ol10 = globals()['B0' + str(i) + '_r_ol10']
    
    P_ol10 = globals()['P0' + str(i) + '_ol10']
    
    W_1_ol10 = 'W0' + str(i) + '_1_ol10'
    W_2_ol10 = 'W0' + str(i) + '_2_ol10'
    
    globals()[W_1_ol10], globals()[W_2_ol10] = projectionMatrix(B_l_ol10, B_r_ol10, P_ol10)


# In[96]:


P01.shape


# In[97]:


W01_1_ol10.shape, W01_2_ol10.shape


# In[98]:


# Print first 10 rows of W1 and W2 should share same result
dframe(W02_1_ol10[:10, :22]).head()


# In[99]:


dframe(W02_2_ol10[:10, :22]).head()


# In[100]:


# Finally just pick one W as projection matrix
# =========== Normal Data ====================
for i in range(1, ns+1):
    W = 'W0' + str(i)
    W_1 = globals()['W0' + str(i) + '_1']
    
    globals()[W] = W_1

    
# =========== 90% Overlap ====================
for i in range(1, ns+1):
    W_ol90 = 'W0' + str(i) + '_ol90'
    W_1_ol90 = globals()['W0' + str(i) + '_1_ol90']
    
    globals()[W_ol90] = W_1_ol90
    
# =========== 50% Overlap ====================
for i in range(1, ns+1):
    W_ol50 = 'W0' + str(i) + '_ol50'
    W_1_ol50 = globals()['W0' + str(i) + '_1_ol50']
    
    globals()[W_ol50] = W_1_ol50


# =========== 10% Overlap ====================
for i in range(1, ns+1):
    W_ol10 = 'W0' + str(i) + '_ol10'
    W_1_ol10 = globals()['W0' + str(i) + '_1_ol10']
    
    globals()[W_ol10] = W_1_ol10


# In[101]:


dframe(W04_ol10).head()


# ### Feature Vector for Training Data
# Finding feature vector $f_p$. The feature vector $f_p$ is given as:
# $$f_p = log(\frac{var(Z_p)}{\sum_{i=1}^{2m}var(Z_i)})$$
# <br>
# where,<br>
# $var(Z_p)$ denotes variance of samples in each row $p$, from 1 to 2m 
# <br>
# $\sum_{i=1}^{2m}var(Z_i)$ denotes sum over all variances of samples in row 1 to 2m

# In[102]:


# Used the projection matrix W from training dataset
W01.shape, W01_ol90.shape, W01_ol50.shape, W01_ol10.shape


# In[103]:


def newW(W, m):
    # So take the first m of first and last row of matrix
    # And remove the rest of matrix row
    x, y = W.shape
    W_new = np.delete(W, np.s_[m+1:x-m+1], 0)
    
    return W_new


# In[104]:


help(newW)


# In[105]:


# Let's follow the procedure in the paper and set m = 2
# So 2m = 4
m = 2
# ------- Normal Data -------------------
for i in range(1, ns+1):
    W = globals()['W0' + str(i)]
    W_new = 'W0' + str(i)
    
    globals()[W_new] = newW(W, 2) 

# ------- 90% Overlapped Data ---------------
    W_ol90 = globals()['W0' + str(i) + '_ol90']
    W_ol90_new = 'W0' + str(i) + '_ol90'

    globals()[W_ol90_new] = newW(W_ol90, 2)

# ------- 50% Overlapped Data ---------------
    W_ol50 = globals()['W0' + str(i) + '_ol50']
    W_ol50_new = 'W0' + str(i) + '_ol50'

    globals()[W_ol50_new] = newW(W_ol50, 2)
    
# ------- 10% Overlapped Data ---------------
    W_ol10 = globals()['W0' + str(i) + '_ol10']
    W_ol10_new = 'W0' + str(i) + '_ol10'

    globals()[W_ol10_new] = newW(W_ol10, 2)


# In[106]:


# # Overlap data
# # Subject 01
# W01_ol90n = newW(W01_ol90, m)

# # Subject 02
# W02_ol90n = newW(W02_ol90, m)


# In[107]:


dframe(W01_ol10)


# In[108]:


# Shape of new W
# print(i)
W01_ol90.shape


# So now we have matrix W with dimension of $2m * N$
# 
# Apply this matrix to original EEG trial signals E with the dim $N*T$ to obtain $Z$ with dim of $2m * T$

# Now calculate Z
# 
#         Z = W * E
#     2m x N = 2m x N * N x T 

# In[109]:


E09_l_ol90_tr.shape, E09_r_ol90_tr.shape


# In[110]:


E09_l_ol50_tr.shape, E09_r_ol50_tr.shape


# In[111]:


E08_l_ol10_tr.shape, E08_r_ol10_tr.shape


# In[112]:


E09_l_tr.shape, E09_r_tr.shape


# In[113]:


def computeZ(E_left, E_right, W):
    '''
    
    '''
    # Tranposing E to dimension of trial x electrodes x samples
    if E_left.shape[1] != 22 :
        E_left = np.transpose(E_left, (0, 2, 1))
    if E_right.shape[1] != 22 :
        E_right = np.transpose(E_left, (0, 2, 1))
        
    # New array of zeros for Z
    Z_left = np.zeros((E_left.shape[0], W.shape[0], E_left.shape[2]))
    Z_right = np.zeros((E_right.shape[0], W.shape[0], E_right.shape[2]))
    
    # For left class
    for i in range(Z_left.shape[0]):
        Z_left[i] = W@E_left[i]

    # For right class
    for i in range(Z_right.shape[0]):
        Z_right[i] = W@E_right[i]
        
    return Z_left, Z_right


# In[114]:


help(computeZ)


# In[115]:


W01_ol90.shape


# In[116]:


# ------- Normal Data -------------------
for i in range(1, ns+1):
    E_l_tr = globals()['E0' + str(i) + '_l_tr']
    E_r_tr = globals()['E0' + str(i) + '_r_tr']
    W = globals()['W0' + str(i)]
    
    Z_l = 'Z0' + str(i) + '_l'
    Z_r = 'Z0' + str(i) + '_r'
    
    globals()[Z_l], globals()[Z_r] = computeZ(E_l_tr, E_r_tr, W)

    
# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    E_l_ol90_tr = globals()['E0' + str(i) + '_l_ol90_tr']
    E_r_ol90_tr = globals()['E0' + str(i) + '_r_ol90_tr']
    W_ol90 = globals()['W0' + str(i) +'_ol90']
    
    Z_l_ol90 = 'Z0' + str(i) + '_l_ol90'
    Z_r_ol90 = 'Z0' + str(i) + '_r_ol90'
    
    globals()[Z_l_ol90], globals()[Z_r_ol90] = computeZ(E_l_ol90_tr, E_r_ol90_tr, W_ol90)


# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    E_l_ol50_tr = globals()['E0' + str(i) + '_l_ol50_tr']
    E_r_ol50_tr = globals()['E0' + str(i) + '_r_ol50_tr']
    W_ol50 = globals()['W0' + str(i) +'_ol50']
    
    Z_l_ol50 = 'Z0' + str(i) + '_l_ol50'
    Z_r_ol50 = 'Z0' + str(i) + '_r_ol50'
    
    globals()[Z_l_ol50], globals()[Z_r_ol50] = computeZ(E_l_ol50_tr, E_r_ol50_tr, W_ol50)


# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    E_l_ol10_tr = globals()['E0' + str(i) + '_l_ol10_tr']
    E_r_ol10_tr = globals()['E0' + str(i) + '_r_ol10_tr']
    W_ol10 = globals()['W0' + str(i) +'_ol10']
    
    Z_l_ol10 = 'Z0' + str(i) + '_l_ol10'
    Z_r_ol10 = 'Z0' + str(i) + '_r_ol10'
    
    globals()[Z_l_ol10], globals()[Z_r_ol10] = computeZ(E_l_ol10_tr, E_r_ol10_tr, W_ol10)


# In[117]:


Z01_l.shape, Z01_r.shape, Z01_l_ol10.shape, Z01_r_ol10.shape


# In[118]:


Z01_l_ol50.shape, Z01_r_ol50.shape, Z01_l_ol90.shape, Z01_r_ol90.shape


# In[119]:


# ===== Feature Vector ===== #
def featVector(Z_left, Z_right):
    # New array of zeros for training feature left and right
    feat_left =  np.zeros((Z_left.shape[0], Z_left.shape[1], 1))
    feat_right =  np.zeros((Z_right.shape[0], Z_right.shape[1], 1))
    
    # For left class
    for i in range(Z_left.shape[0]):
        var = np.var(Z_left[i], ddof=1, axis=1)[:, np.newaxis]
        varsum = np.sum(var)
        feat_left[i] = (var/varsum);

    # For right class
    for i in range(Z_right.shape[0]):
        var = np.var(Z_right[i], ddof=1, axis=1)[:, np.newaxis]
        varsum = np.sum(var)
        feat_right[i] = (var/varsum);
        
    return feat_left, feat_right


# In[120]:


# ------- Normal Data -------------------
for i in range(1, ns+1):
    Z_l = globals()['Z0' + str(i) + '_l']
    Z_r = globals()['Z0' + str(i) + '_r']
    
    feat_l_tr = 'feat0' + str(i) + '_l_tr'
    feat_r_tr = 'feat0' + str(i) + '_r_tr'

    globals()[feat_l_tr], globals()[feat_r_tr] = featVector(Z_l, Z_r)    
    
# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    Z_l_ol90 = globals()['Z0' + str(i) + '_l_ol90']
    Z_r_ol90 = globals()['Z0' + str(i) + '_r_ol90']
    
    feat_l_ol90_tr = 'feat0' + str(i) + '_l_ol90_tr'
    feat_r_ol90_tr = 'feat0' + str(i) + '_r_ol90_tr'

    globals()[feat_l_ol90_tr], globals()[feat_r_ol90_tr] = featVector(Z_l_ol90, Z_r_ol90)
    

# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    Z_l_ol50 = globals()['Z0' + str(i) + '_l_ol50']
    Z_r_ol50 = globals()['Z0' + str(i) + '_r_ol50']
    
    feat_l_ol50_tr = 'feat0' + str(i) + '_l_ol50_tr'
    feat_r_ol50_tr = 'feat0' + str(i) + '_r_ol50_tr'

    globals()[feat_l_ol50_tr], globals()[feat_r_ol50_tr] = featVector(Z_l_ol50, Z_r_ol50)  


# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    Z_l_ol10 = globals()['Z0' + str(i) + '_l_ol10']
    Z_r_ol10 = globals()['Z0' + str(i) + '_r_ol10']
    
    feat_l_ol10_tr = 'feat0' + str(i) + '_l_ol10_tr'
    feat_r_ol10_tr = 'feat0' + str(i) + '_r_ol10_tr'

    globals()[feat_l_ol10_tr], globals()[feat_r_ol10_tr] = featVector(Z_l_ol10, Z_r_ol10)    


# In[121]:


feat09_l_ol90_tr.shape, feat09_r_ol90_tr.shape, feat09_l_ol50_tr.shape, feat09_r_ol50_tr.shape


# In[122]:


feat09_l_ol10_tr.shape, feat09_r_ol10_tr.shape, feat09_l_tr.shape, feat09_r_tr.shape


# In[123]:


# Adding corresponding label column to feature vector, then shuffle
def addLabel(feat_left, feat_right):
    # Reshape feature vector matrix to 2D
    feat_left = np.reshape(feat_left, (feat_left.shape[0], feat_left.shape[1]))
    feat_right = np.reshape(feat_right, (feat_right.shape[0], feat_right.shape[1]))
    
    # Label one for left and zero for right
    label_one = np.ones((feat_left.shape[0], 1))
    label_zero = label_one * 0
    
    # Add label to associated class
    feat_left = np.hstack([feat_left, label_one])
    feat_right = np.hstack([feat_right, label_zero])
    
    # Vertically stack them
    feat = np.vstack([feat_left, feat_right])
    
    # Shuffles them
    np.random.shuffle(feat)
    
    return feat


# In[124]:


# Adding labels
# ------- Normal Data -------------------
for i in range(1, ns+1):
    feat_l_tr = globals()['feat0' + str(i) + '_l_tr']
    feat_r_tr = globals()['feat0' + str(i) + '_r_tr']
    
    feat_tr = 'feat0' + str(i) + '_tr'
    globals()[feat_tr] = addLabel(feat_l_tr, feat_r_tr)


# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    feat_l_ol90_tr = globals()['feat0' + str(i) + '_l_ol90_tr']
    feat_r_ol90_tr = globals()['feat0' + str(i) + '_r_ol90_tr']
    
    feat_ol90_tr = 'feat0' + str(i) + '_ol90_tr'
    globals()[feat_ol90_tr] = addLabel(feat_l_ol90_tr, feat_r_ol90_tr)
    
    
# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    feat_l_ol50_tr = globals()['feat0' + str(i) + '_l_ol50_tr']
    feat_r_ol50_tr = globals()['feat0' + str(i) + '_r_ol50_tr']
    
    feat_ol50_tr = 'feat0' + str(i) + '_ol50_tr'
    globals()[feat_ol50_tr] = addLabel(feat_l_ol50_tr, feat_r_ol50_tr)
    
    
# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    feat_l_ol10_tr = globals()['feat0' + str(i) + '_l_ol10_tr']
    feat_r_ol10_tr = globals()['feat0' + str(i) + '_r_ol10_tr']
    
    feat_ol10_tr = 'feat0' + str(i) + '_ol10_tr'
    globals()[feat_ol10_tr] = addLabel(feat_l_ol10_tr, feat_r_ol10_tr)


# In[125]:


type(feat01_tr)


# In[126]:


feat09_ol90_tr.shape, feat01_ol50_tr.shape


# In[127]:


feat01_ol10_tr.shape, feat01_tr.shape


# In[128]:


dframe(feat01_tr, columns=['feat1', 'feat2', 'feat3', 'feat4', 'label'])


# ## (Skipped) Perform CAR(common average reference) to dataset

# In[ ]:


# print('Perform CAR in each class, left and right. Executing...')
# CAR value of left class
# avgleft = np.mean(E_left, axis=2)[:, :, np.newaxis]
# CAR_left = E_left - avgleft;

# CAR value of right class
# avgright = np.mean(E_right, axis=2)[:, :, np.newaxis]
# CAR_right = E_right - avgright;


# In[ ]:


# Don't compute the CAR, thus


# In[ ]:


# pd.DataFrame(CAR_left[48, :, :], columns=Xcol)


# In[ ]:


# CAR_left.shape, CAR_right.shape


# ## Processing Test Data

# <b><i> During processing of test data, projection matrix W doesnt need to be computed, instead,
# <br>W from training data is used to compute feature vector for test dataset </i></b>

# ### Feature Vector for Test Data
# Finding feature vector $f_p$. The feature vector $f_p$ is given as:
# $$f_p = log(\frac{var(Z_p)}{\sum_{i=1}^{2m}var(Z_i)})$$
# <br>
# where,<br>
# $var(Z_p)$ denotes variance of samples in each row $p$, from 1 to 2m 
# <br>
# $\sum_{i=1}^{2m}var(Z_i)$ denotes sum over all variances of samples in row 1 to 2m

# In[129]:


# W from training data
W01.shape


# So now we have matrix W with dimension of $2m * N$
# 
# Apply this matrix to original EEG trial signals E with the dim $N*T$ to obtain $Z$ with dim of $2m * T$

# Now calculate Z
# 
#         Z = W * E
#     2m x N = 2m x N * N x T 

# In[130]:


E01_l_te.shape, E01_r_te.shape, E01_l_ol90_te.shape, E01_r_ol90_te.shape


# In[131]:


help(computeZ)


# In[132]:


# ------- Normal Data -------------------
for i in range(1, ns+1):
    E_l_te = globals()['E0' + str(i) + '_l_te']
    E_r_te = globals()['E0' + str(i) + '_r_te']
    W = globals()['W0' + str(i)]
    
    Z_l_te = 'Z0' + str(i) + '_l_te'
    Z_r_te = 'Z0' + str(i) + '_r_te'
    
    globals()[Z_l_te], globals()[Z_r_te] = computeZ(E_l_te, E_r_te, W)

    
# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    E_l_ol90_te = globals()['E0' + str(i) + '_l_ol90_te']
    E_r_ol90_te = globals()['E0' + str(i) + '_r_ol90_te']
    W_ol90 = globals()['W0' + str(i) +'_ol90']
    
    Z_l_ol90_te = 'Z0' + str(i) + '_l_ol90_te'
    Z_r_ol90_te = 'Z0' + str(i) + '_r_ol90_te'
    
    globals()[Z_l_ol90_te], globals()[Z_r_ol90_te] = computeZ(E_l_ol90_te, E_r_ol90_te, W_ol90)
    

# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    E_l_ol50_te = globals()['E0' + str(i) + '_l_ol50_te']
    E_r_ol50_te = globals()['E0' + str(i) + '_r_ol50_te']
    W_ol50 = globals()['W0' + str(i) +'_ol50']
    
    Z_l_ol50_te = 'Z0' + str(i) + '_l_ol50_te'
    Z_r_ol50_te = 'Z0' + str(i) + '_r_ol50_te'
    
    globals()[Z_l_ol50_te], globals()[Z_r_ol50_te] = computeZ(E_l_ol50_te, E_r_ol50_te, W_ol50)
    
# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    E_l_ol10_te = globals()['E0' + str(i) + '_l_ol10_te']
    E_r_ol10_te = globals()['E0' + str(i) + '_r_ol10_te']
    W_ol10 = globals()['W0' + str(i) +'_ol10']
    
    Z_l_ol10_te = 'Z0' + str(i) + '_l_ol10_te'
    Z_r_ol10_te = 'Z0' + str(i) + '_r_ol10_te'
    
    globals()[Z_l_ol10_te], globals()[Z_r_ol10_te] = computeZ(E_l_ol10_te, E_r_ol10_te, W_ol10)


# In[133]:


Z01_l_te.shape, Z01_r_te.shape, Z01_l_ol90_te.shape, Z01_r_ol90_te.shape


# In[134]:


help(featVector)


# In[135]:


# ===== Feature Vector ===== #
def featVector(Z_left, Z_right):
    # New array of zeros for training feature left and right
    feat_left =  np.zeros((Z_left.shape[0], Z_left.shape[1], 1))
    feat_right =  np.zeros((Z_right.shape[0], Z_right.shape[1], 1))
    
    # For left class
    for i in range(Z_left.shape[0]):
        var = np.var(Z_left[i], ddof=1, axis=1)[:, np.newaxis]
        varsum = np.sum(var)
        feat_left[i] = (var/varsum);

    # For right class
    for i in range(Z_right.shape[0]):
        var = np.var(Z_right[i], ddof=1, axis=1)[:, np.newaxis]
        varsum = np.sum(var)
        feat_right[i] = (var/varsum);
        
    return feat_left, feat_right


# In[136]:


# ------- Normal Data -------------------
for i in range(1, ns+1):
    Z_l_te = globals()['Z0' + str(i) + '_l_te']
    Z_r_te = globals()['Z0' + str(i) + '_r_te']
    
    feat_l_te = 'feat0' + str(i) + '_l_te'
    feat_r_te = 'feat0' + str(i) + '_r_te'

    globals()[feat_l_te], globals()[feat_r_te] = featVector(Z_l_te, Z_r_te)    
    
# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    Z_l_ol90_te = globals()['Z0' + str(i) + '_l_ol90_te']
    Z_r_ol90_te = globals()['Z0' + str(i) + '_r_ol90_te']
    
    feat_l_ol90_te = 'feat0' + str(i) + '_l_ol90_te'
    feat_r_ol90_te = 'feat0' + str(i) + '_r_ol90_te'

    globals()[feat_l_ol90_te], globals()[feat_r_ol90_te] = featVector(Z_l_ol90_te, Z_r_ol90_te)    

# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    Z_l_ol50_te = globals()['Z0' + str(i) + '_l_ol50_te']
    Z_r_ol50_te = globals()['Z0' + str(i) + '_r_ol50_te']
    
    feat_l_ol50_te = 'feat0' + str(i) + '_l_ol50_te'
    feat_r_ol50_te = 'feat0' + str(i) + '_r_ol50_te'

    globals()[feat_l_ol50_te], globals()[feat_r_ol50_te] = featVector(Z_l_ol50_te, Z_r_ol50_te)
    
# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    Z_l_ol10_te = globals()['Z0' + str(i) + '_l_ol10_te']
    Z_r_ol10_te = globals()['Z0' + str(i) + '_r_ol10_te']
    
    feat_l_ol10_te = 'feat0' + str(i) + '_l_ol10_te'
    feat_r_ol10_te = 'feat0' + str(i) + '_r_ol10_te'

    globals()[feat_l_ol10_te], globals()[feat_r_ol10_te] = featVector(Z_l_ol10_te, Z_r_ol10_te)  


# In[137]:


feat01_l_ol90_te.shape, feat01_r_ol90_te.shape, feat01_l_ol50_te.shape, feat01_r_ol50_te.shape


# In[138]:


feat01_l_ol10_te.shape, feat01_r_ol10_te.shape, feat01_r_te.shape, feat01_l_te.shape, 


# In[139]:


help(addLabel)


# In[140]:


# ------- Normal Data -------------------
for i in range(1, ns+1):
    feat_l_te = globals()['feat0' + str(i) + '_l_te']
    feat_r_te = globals()['feat0' + str(i) + '_r_te']
    
    feat_te = 'feat0' + str(i) + '_te'
    globals()[feat_te] = addLabel(feat_l_te, feat_r_te)


# ------- 90% Overlapped Data ---------------
for i in range(1, ns+1):
    feat_l_ol90_te = globals()['feat0' + str(i) + '_l_ol90_te']
    feat_r_ol90_te = globals()['feat0' + str(i) + '_r_ol90_te']
    
    feat_ol90_te = 'feat0' + str(i) + '_ol90_te'
    globals()[feat_ol90_te] = addLabel(feat_l_ol90_te, feat_r_ol90_te)
    

# ------- 50% Overlapped Data ---------------
for i in range(1, ns+1):
    feat_l_ol50_te = globals()['feat0' + str(i) + '_l_ol50_te']
    feat_r_ol50_te = globals()['feat0' + str(i) + '_r_ol50_te']
    
    feat_ol50_te = 'feat0' + str(i) + '_ol50_te'
    globals()[feat_ol50_te] = addLabel(feat_l_ol50_te, feat_r_ol50_te)
    

# ------- 10% Overlapped Data ---------------
for i in range(1, ns+1):
    feat_l_ol10_te = globals()['feat0' + str(i) + '_l_ol10_te']
    feat_r_ol10_te = globals()['feat0' + str(i) + '_r_ol10_te']
    
    feat_ol10_te = 'feat0' + str(i) + '_ol10_te'
    globals()[feat_ol10_te] = addLabel(feat_l_ol10_te, feat_r_ol10_te)


# In[141]:


type(feat01_te)


# In[142]:


feat01_ol90_te.shape, feat01_ol50_te.shape,


# In[143]:


feat01_ol10_te.shape, feat01_te.shape


# In[144]:


dframe(feat03_ol50_te, columns=['feat1', 'feat2', 'feat3', 'feat4', 'label']).head()


# ## Classification

# ### Preparing X train, X test, y train, y test 

# #### Normal Data

# In[145]:


# ======= Train data & Test Data ============
# Get X train, y train, X test, y test
# Normal data, 
for i in range(1, ns+1):
    feat_tr = globals()['feat0' + str(i) + '_tr']
    feat_te = globals()['feat0' + str(i) + '_te']

    X_tr = 'X0' + str(i) + '_tr'
    y_tr = 'y0' + str(i) + '_tr'
    
    X_te = 'X0' + str(i) + '_te'
    y_te = 'y0' + str(i) + '_te'
    
    globals()[X_tr] = feat_tr[:, :4]
    globals()[y_tr] = feat_tr[:, 4]
    
    globals()[X_te] = feat_te[:, :4]
    globals()[y_te] = feat_te[:, 4]


# In[146]:


X01_tr.shape, y01_tr.shape


# In[147]:


X01_te.shape, y01_te.shape


# #### Overlap 90%

# In[148]:


# ======= Train data & Test Data ============
# Get X train, y train, X test, y test
# 90% Overlap data, 
for i in range(1, ns+1):
    feat_ol90_tr = globals()['feat0' + str(i) + '_ol90_tr']
    feat_ol90_te = globals()['feat0' + str(i) + '_ol90_te']

    X_ol90_tr = 'X0' + str(i) + '_ol90_tr'
    y_ol90_tr = 'y0' + str(i) + '_ol90_tr'
    
    X_ol90_te = 'X0' + str(i) + '_ol90_te'
    y_ol90_te = 'y0' + str(i) + '_ol90_te'
    
    globals()[X_ol90_tr] = feat_ol90_tr[:, :4]
    globals()[y_ol90_tr] = feat_ol90_tr[:, 4]
    
    globals()[X_ol90_te] = feat_ol90_te[:, :4]
    globals()[y_ol90_te] = feat_ol90_te[:, 4]


# In[149]:


feat09_ol90_tr.shape


# In[150]:


feat09_ol90_te.shape


# In[151]:


X09_ol90_tr.shape, X09_ol90_tr.shape


# In[152]:


X09_ol90_te.shape, y09_ol90_te.shape


# #### Overlap 50%

# In[153]:


# ======= Train data & Test Data ============
# Get X train, y train, X test, y test
# 50% Overlap data, 
for i in range(1, ns+1):
    feat_ol50_tr = globals()['feat0' + str(i) + '_ol50_tr']
    feat_ol50_te = globals()['feat0' + str(i) + '_ol50_te']

    X_ol50_tr = 'X0' + str(i) + '_ol50_tr'
    y_ol50_tr = 'y0' + str(i) + '_ol50_tr'
    
    X_ol50_te = 'X0' + str(i) + '_ol50_te'
    y_ol50_te = 'y0' + str(i) + '_ol50_te'
    
    globals()[X_ol50_tr] = feat_ol50_tr[:, :4]
    globals()[y_ol50_tr] = feat_ol50_tr[:, 4]
    
    globals()[X_ol50_te] = feat_ol50_te[:, :4]
    globals()[y_ol50_te] = feat_ol50_te[:, 4]


# In[154]:


X05_ol50_tr.shape, X05_ol50_tr.shape


# In[155]:


X05_ol50_te.shape, y05_ol50_te.shape


# #### Overlap 10%

# In[156]:


# ======= Train data & Test Data ============
# Get X train, y train, X test, y test
# 10% Overlap data, 
for i in range(1, ns+1):
    feat_ol10_tr = globals()['feat0' + str(i) + '_ol10_tr']
    feat_ol10_te = globals()['feat0' + str(i) + '_ol10_te']

    X_ol10_tr = 'X0' + str(i) + '_ol10_tr'
    y_ol10_tr = 'y0' + str(i) + '_ol10_tr'
    
    X_ol10_te = 'X0' + str(i) + '_ol10_te'
    y_ol10_te = 'y0' + str(i) + '_ol10_te'
    
    globals()[X_ol10_tr] = feat_ol10_tr[:, :4]
    globals()[y_ol10_tr] = feat_ol10_tr[:, 4]
    
    globals()[X_ol10_te] = feat_ol10_te[:, :4]
    globals()[y_ol10_te] = feat_ol10_te[:, 4]


# In[157]:


X09_ol10_tr.shape, y09_ol10_tr.shape


# In[158]:


X01_ol10_te.shape, y01_ol10_te.shape


# ### Verify the SVM Classifier
# By applying the train model into train data

# In[159]:


from sklearn import svm
from sklearn import metrics


# In[160]:


model = svm.SVC()


# In[161]:


model.fit(X01_tr, y01_tr)


# In[162]:


ypred = model.predict(X01_te)


# In[163]:


ypred.shape


# In[164]:


acc = metrics.accuracy_score(ypred, y01_te)
acc


# In[165]:


ypred = model.predict(X01_tr)
acc = metrics.accuracy_score(ypred, y01_tr)
acc


# In[ ]:





# In[ ]:





# ###  Calling SVM classifier

# In[166]:


from sklearn import svm
from sklearn import metrics
svc_model = svm.SVC()


# In[167]:


'''
Function that will evaluate train and test data over iteration of C value
'''
def evaluate(X_train, y_train, X_test, y_test):
    maxC = 100
    inc = 10 
    temp = []
    
    for C_val in range(0, maxC+1, inc):
        if C_val == 0:
            svc_model.C = C_val+1
        else:
            svc_model.C = C_val

        svc_model.fit(X_train, y_train) # Model training
        y_pred = svc_model.predict(X_test) # Model prediction
        acc = metrics.accuracy_score(y_test, y_pred)
        acc *= 100
        
        temp += [svc_model.C, acc]
        res = np.array(temp).reshape(-1, 2)
    return res


# ### Evaluating Model (Accuracy, Precision, Recall etc.)

# In[168]:


# ========= Evaluating Model =============
# Normal data
for i in range(1, ns+1):
    X_tr = globals()['X0' + str(i) + '_tr']
    y_tr = globals()['y0' + str(i) + '_tr']
    
    X_te = globals()['X0' + str(i) + '_te']
    y_te = globals()['y0' + str(i) + '_te']
    
    res = 'res0' + str(i)
    
    globals()[res] = evaluate(X_tr, y_tr, X_te, y_te)

# 90% Overlap data
    X_ol90_tr = globals()['X0' + str(i) + '_ol90_tr']
    y_ol90_tr = globals()['y0' + str(i) + '_ol90_tr']
    
    X_ol90_te = globals()['X0' + str(i) + '_ol90_te']
    y_ol90_te = globals()['y0' + str(i) + '_ol90_te']
    
    res_ol90 = 'res0' + str(i) + '_ol90'
    
    globals()[res_ol90] = evaluate(X_ol90_tr, y_ol90_tr, X_ol90_te, y_ol90_te)
    
# 50% Overlap data
    X_ol50_tr = globals()['X0' + str(i) + '_ol50_tr']
    y_ol50_tr = globals()['y0' + str(i) + '_ol50_tr']
    
    X_ol50_te = globals()['X0' + str(i) + '_ol50_te']
    y_ol50_te = globals()['y0' + str(i) + '_ol50_te']
    
    res_ol50 = 'res0' + str(i) + '_ol50'
    
    globals()[res_ol50] = evaluate(X_ol50_tr, y_ol50_tr, X_ol50_te, y_ol50_te)
    
# 10% Overlap data
    X_ol10_tr = globals()['X0' + str(i) + '_ol10_tr']
    y_ol10_tr = globals()['y0' + str(i) + '_ol10_tr']
    
    X_ol10_te = globals()['X0' + str(i) + '_ol10_te']
    y_ol10_te = globals()['y0' + str(i) + '_ol10_te']
    
    res_ol10 = 'res0' + str(i) + '_ol10'
    
    globals()[res_ol10] = evaluate(X_ol10_tr, y_ol10_tr, X_ol10_te, y_ol10_te)


# In[169]:


# # Subject 01
# res01n = evaluate(X01_tr, y01_tr, X01_te, y01_te)

# # Subject 02
# res02n = evaluate(X02_tr, y02_tr, X02_te, y02_te)


# In[170]:


# Normal data
cols = ['C Value', 'Accuracy']
dframe(res09, columns=cols)


# ## Plot C vs Accuracy

# In[171]:


# Index for x-axis
idx = np.arange(len(res01[:, 0]))

# All data share common C val
Cval = res01[:, 0]


# In[172]:


# Function to show all ten subject figures
def showFig(ol, idx, CVal):
    i=1

    fig, ax = plt.subplots(2, 5, figsize=(25, 10))
        
    if ol==None:    
        fig.suptitle('C vs Accuracy Original Data', fontsize=25)
    
        for j in range(ax.shape[0]):
            for k in range(ax.shape[1]):
                res = globals()['res0' + str(i)]
                co = (0, 0.3, np.random.uniform(0.2, 1))
                ax[j, k].bar(idx, res[:, 1], color=co)
                ax[j, k].set_title('Subject' + str(i), fontsize=17)
                ax[j, k].set_xticks(idx)
                ax[j, k].set_xticklabels(Cval, rotation=40, fontsize=12)

                ax[j, k].set_yticklabels(np.arange(0, 101, 20), fontsize=15)
                ax[j, k].set_ylim(0, 100)

                if (i<ns):
                    i += 1

        plt.savefig('normal-C-acc.png')
        fig.delaxes(ax[1, 4])
    
    else:
        fig.suptitle('C vs ' + str(ol) + '% Overlap ' + 'Accuracy Data', fontsize=25)

        for j in range(ax.shape[0]):
            for k in range(ax.shape[1]):
                res = globals()['res0' + str(i) + '_ol' + str(ol)]
                    
                if ol == 90:
                    co = (0.8, np.random.uniform(0.1, 0.5), 0.2)
                elif ol == 50:
                    co = (np.random.uniform(0.1, 0.7), 0.5, 0.2)
                else:
                    co = (0.6, np.random.uniform(0.2, 0.6), 0.3)

                
                ax[j, k].bar(idx, res[:, 1], color=co)
                ax[j, k].set_title('Subject' + str(i), fontsize=17)
                ax[j, k].set_xticks(idx)
                ax[j, k].set_xticklabels(Cval, rotation=40, fontsize=12)

                ax[j, k].set_yticklabels(np.arange(0, 101, 20), fontsize=15)
                ax[j, k].set_ylim(0, 100)

                if (i<ns):
                    i += 1

        plt.savefig(str(ol) + '-Overlap' + '-C-acc.png')
        fig.delaxes(ax[1, 4])


# In[173]:


# ============== Original Data ====================
showFig(ol=None, idx=idx, CVal=Cval)

# ============== 90% Overlap Data =================
showFig(ol=90, idx=idx, CVal=Cval)

# ============== 50% Overlap Data =================
showFig(ol=50, idx=idx, CVal=Cval)

# ============== 10% Overlap Data ==================
showFig(ol=10, idx=idx, CVal=Cval)


# In[174]:


# ================ Comparing =====================

no = input('Which subject to compare? (1-9):')
res = globals()['res0' + str(no)]
res_ol10 = globals()['res0' + str(no) + '_ol10']
res_ol50 = globals()['res0' + str(no) + '_ol50']
res_ol90 = globals()['res0' + str(no) + '_ol90']


# ======== Data 1 ========
fig_comp, ax = plt.subplots(1, 4, sharey = True, figsize=(20, 5))
fig_comp.suptitle('C value vs Accuracy', fontsize=25)

co1 = (0, 0.3, np.random.uniform(0.2, 1))

ax[0].bar(idx, res[:, 1], color=co1)
ax[0].set_title('Original', fontsize=20)
ax[0].set_xticks(idx)
ax[0].set_xticklabels(Cval, rotation=40, fontsize=15)

ax[0].set_ylim(0, 100)
ax[0].set_yticklabels(np.arange(0, 101, 20), fontsize=15)

# ======== Data 2 ========
co2 = (0.6, np.random.uniform(0.2, 0.6), 0.3)

ax[1].bar(idx, res_ol10[:, 1], color=co2)
ax[1].set_title('Overlap 10', fontsize=20)
ax[1].set_xticks(idx)
ax[1].set_xticklabels(Cval, rotation=40, fontsize=15)

ax[1].set_ylim(0, 100)
ax[1].set_yticklabels(np.arange(0, 101, 20), fontsize=15)

# ======== Data 3 ==========

co3 = (np.random.uniform(0.1, 0.7), 0.5, 0.2)

ax[2].bar(idx, res_ol50[:, 1], color=co3)
ax[2].set_title('Overlap 50', fontsize=20)
ax[2].set_xticks(idx)
ax[2].set_xticklabels(Cval, rotation=40, fontsize=15)

ax[2].set_ylim(0, 100)
ax[2].set_yticklabels(np.arange(0, 101, 20), fontsize=15)

# ======== Data 4 ======== 
co4 = (0.8, np.random.uniform(0.1, 0.5), 0.2) # Ramdomize color

ax[3].bar(idx, res_ol90[:, 1], color=co4)
ax[3].set_title('Overlap 90', fontsize=20)
ax[3].set_xticks(idx)
ax[3].set_xticklabels(Cval, rotation=40, fontsize=15)

ax[3].set_ylim(0, 100)
ax[3].set_yticklabels(np.arange(0, 101, 20), fontsize=15)



plt.show()


# ## Signal Visualization

# Step:  
# 1. Grab the number of samples shown in figure below in each trial 
# 2. Store them into left and right array for each subject, each containing C3 and C4
# 3. Average over all trials
# 4. Plot the signal

# ![vis-eeg-clip1](./img/vis-eeg-clip.png)

# In[175]:


fs, ns, cue_dur, mi_dur


# In[ ]:


print('The amount of samples to be clipped:', mi_dur+cue_dur+125)


# In[217]:


# # Fetch 1126 samples of each event from sample_data
# # Store them in E_left and E_right respectively, For convenience 'E' is used to imitate the variable used in paper
# # Preparing the variables to store event samples of shape 72 x 1126 x 22
# # Subject 01-09
# for i in range(1, 10):
#     E_l_vis = 'E0' + str(i) + '_l_vis'
#     E_r_vis = 'E0' + str(i) + '_r_vis'
    
#     pos_l = globals()['pos0' + str(i) + '_l'] # Contains etype, index of samples coresponds to left
#     pos_r = globals()['pos0' + str(i) + '_r'] # Contains etype, index of samples coresponds to right
    
#     sample = globals()['sample_data0' + str(i)]
    
#     globals()[E_l_vis] = np.zeros([pos_l.shape[0], mi_dur+cue_dur+125, sample.shape[1]])
#     globals()[E_r_vis] = np.zeros([pos_r.shape[0], mi_dur+cue_dur+125, sample.shape[1]])


# In[212]:


# Fetch 1126 samples of each event from sample_data
# Now we are ready to take 1126 samples of left and right
# Subject 01-09
for i in range(1, 10):    
    E_l_vis = 'E0' + str(i) + '_l_vis'
    E_r_vis = 'E0' + str(i) + '_r_vis'
    
    # This will initialize array shape trial x sample x electrode
    globals()[E_l_vis] = np.zeros([pos_l.shape[0], mi_dur+cue_dur+125, sample.shape[1]])
    globals()[E_r_vis] = np.zeros([pos_r.shape[0], mi_dur+cue_dur+125, sample.shape[1]])
    
    sample = globals()['sample_data0' + str(i)]

    pos_l = globals()['pos0' + str(i) + '_l']
    pos_r = globals()['pos0' + str(i) + '_r']
    
    for j in range(len(pos_l)):
        globals()[E_l_vis][j] = sample.iloc[pos_l[j]-125 : pos_l[j]+cue_dur+mi_dur, :]
        
    for k in range(len(pos_l)):
        globals()[E_r_vis][k] = sample.iloc[pos_r[j]-125 : pos_r[j]+cue_dur+mi_dur, :]


# In[219]:


E05_l_vis.shape, E05_r_vis.shape 


# In[220]:


# Convert them to N x T array, N = electrodes, T = samples
for i in range(1, ns+1):
    E_l = globals()['E0' + str(i) + '_l_vis']
    E_r = globals()['E0' + str(i) + '_r_vis']
    
    E_l_trans = 'E0' + str(i) + '_l_vis'
    E_r_trans = 'E0' + str(i) + '_r_vis'

    if (E_l.shape[1] != 22 and E_r.shape[1] != 22):
        globals()[E_l_trans] = np.transpose(E_l, (0, 2, 1))
        globals()[E_r_trans] = np.transpose(E_r, (0, 2, 1))


# In[222]:


E01_l_vis.shape, E01_r_vis.shape


# ### Divide into left class and right class, containing C3 and C4

# In[223]:


# Diving them by left and right class
# So left and right class will be array of one subject containing all trials consist of C3 and C4
# Dimension of array trial x electrodes x samples

# Preparing array
for i in range(1, ns+1):
    left_vis = 'left0' + str(i) + '_vis'
    right_vis = 'right0' + str(i) + '_vis'
    
    E01_l_vis = globals()['E0' + str(i) + '_l_vis']
    E01_r_vis = globals()['E0' + str(i) + '_r_vis']
    
    globals()[left_vis] = np.zeros([E01_l_vis.shape[0], 2, E01_l_vis.shape[2]])
    globals()[right_vis] = np.zeros([E01_r_vis.shape[0], 2, E01_r_vis.shape[2]])
    
# Grab necessary index 
for i in range(1, ns+1):
    left_vis = globals()['left0' + str(i) + '_vis']
    right_vis = globals()['right0' + str(i) + '_vis']
    
    E01_l_vis = globals()['E0' + str(i) + '_l_vis']
    E01_r_vis = globals()['E0' + str(i) + '_r_vis']

    for j in range(E01_l_vis.shape[0]):
        left_vis[j] = np.vstack((E01_l_vis[j, 7, :], E01_l_vis[j, 11, :]))
        right_vis[j] = np.vstack((E01_r_vis[j, 7, :], E01_r_vis[j, 11, :])) 


# In[224]:


left01_vis.shape, right01_vis.shape


# In[226]:


dframe(left01_vis[0], index=['C3', 'C4'])


# ### Averaging over all trials

# In[228]:


# Try averaging over all trials within subject
# Expected array per subject 2 x 1126

for i in range(1, ns+1):
    left_vis_avg = 'left0' + str(i) +'_vis_avg'
    right_vis_avg = 'right0' + str(i) + '_vis_avg'
    
    left_vis = globals()['left0' + str(i) + '_vis']
    right_vis = globals()['right0' + str(i) + '_vis']    
    
    globals()[left_vis_avg] = np.mean(left_vis, axis=0)
    globals()[right_vis_avg] = np.mean(right_vis, axis=0)    


# In[229]:


left01_vis_avg.shape, right01_vis_avg.shape


# In[231]:


dframe(left03_vis_avg, index=['C3', 'C4'])


# ### Plotting Data

# In[ ]:


# fig = plt.figure(figsize=(8, 6))
# plt.plot(left04_avg[0])
# plt.show()


# In[233]:


# Create ticks for x axis
idx = np.arange(len(left01_vis_avg[0]))

# Convert sample to time, divide by fs
xax = idx/fs


# In[234]:


# import pylab as pl
no = input('Which subject (1...9)? ')
side = input('Which side (left/right)? ')

data = globals()[side + '0' + str(no) + '_vis_avg']
figure = plt.figure(figsize = (15, 9))

# ================== C3 =========================
ax1 = figure.add_subplot(221)
ax1.set_title('C3')
ax1.plot(xax, data[0])
ax1.plot([0.5]*50, range(-10, 40), linewidth = 3, c='r', linestyle='dashed')
ax1.plot([1.5]*50, range(-10, 40), 'g--', linewidth = 3)

ax1.set_xticks(np.arange(0, 5, 0.5))
ax1.axis(ymin=-10,ymax=30)

ax1.annotate('Cue Start', xy=(0.5, 10),  xycoords='data',
            xytext=(1.2, 12),
            arrowprops=dict(facecolor='black', width=1),
            horizontalalignment='right', verticalalignment='top',
            size=12)
ax1.annotate('MI Start', xy=(1.5, 10),  xycoords='data',
            xytext=(2.3, 12),
            arrowprops=dict(facecolor='black', width=1),
            horizontalalignment='right', verticalalignment='top',
            size=12)
ax1.set_xlabel('Second')
ax1.set_ylabel('uV?')

ax2 = figure.add_subplot(223)
p = 20*np.log10(np.abs(np.fft.rfft(data[0])))
f = np.linspace(0, fs/2, len(p))

ax2.plot(f, p)
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Power')


# =================== C4 ==========================
ax1 = figure.add_subplot(222)
ax1.set_title('C4')
ax1.plot(xax, data[1])
ax1.plot([0.5]*50, range(-10, 40), linewidth = 3, c='r', linestyle='dashed')
ax1.plot([1.5]*50, range(-10, 40), 'g--', linewidth = 3)

ax1.set_xticks(np.arange(0, 5, 0.5))
ax1.axis(ymin=-10,ymax=30)

ax1.annotate('Cue Start', xy=(0.5, 10),  xycoords='data',
            xytext=(1.2, 12),
            arrowprops=dict(facecolor='black', width=1),
            horizontalalignment='right', verticalalignment='top',
            size=12)
ax1.annotate('MI Start', xy=(1.5, 10),  xycoords='data',
            xytext=(2.3, 12),
            arrowprops=dict(facecolor='black', width=1),
            horizontalalignment='right', verticalalignment='top',
            size=12)
ax1.set_xlabel('Second')
ax1.set_ylabel('uV?')

ax2 = figure.add_subplot(224)
p = 20*np.log10(np.abs(np.fft.rfft(data[1])))
f = np.linspace(0, fs/2, len(p))

ax2.plot(f, p)
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Power')

plt.show()


# ## PSD
# Step
# 1. Check the previous array containing trial data with dimension trial x no. electrodes x samples
# 2. Grab `n` samples from C3 and C4 only, now array shape should be trial x 2 x samples
# 3. Average over all trials
# 4. Find out how to calculate the PSD 
# 4. Create two figures C3 and C4 of each subject
# 5. On one figure, plot left and right class
# 
# 6. Should I average over all trials?
# 
# `n` = duration of MI task
# 
# ![montage](./img/montage.png)
# ![psd-eeg-clip](./img/psd-eeg-clip.png)

# In[237]:


# The trial array from previous computation are E_left and E_right
# This data already contains MI_task samples
E01_left.shape, E01_right.shape


# ### Divide C3 and C4 each subject, containing left and right
# According to the datasheet, electrodes number 8 and 12 corespond to electrodes C3 and C4.   
# Grab electrodes number 8 and 12 from data E_left and E_right.  
#   
# *Note that in python index starts at 0, thus 8 = 7, 12 = 11*

# In[238]:


# Grab index 7 and 11 from each E data
# Create array to store data
for i in range(1, ns+1):
    C3 = 'C3_0' + str(i)
    C4 = 'C4_0' + str(i)
    
    E = globals()['E01_left'] # Both have same size, enough to choose one
    
    globals()[C3] = np.zeros([E.shape[0], 2, E.shape[2]])
    globals()[C4] = np.zeros([E.shape[0], 2, E.shape[2]])

for i in range(1, ns+1):
    C3 = globals()['C3_0' + str(i)]
    C4 = globals()['C4_0' + str(i)]
    
    E_l = globals()['E0' + str(i) + '_left']
    E_r = globals()['E0' + str(i) + '_right']
    
    for j in range(E_l.shape[0]):
        C3[j] = np.vstack((E_l[j, 7, :], E_r[j, 7, :]))
        C4[j] = np.vstack((E_l[j, 11, :], E_r[j, 11, :])) 


# In[ ]:


C3_01.shape, C4


# In[239]:


dframe(C3_01[0], index=['left', 'right'])


# ### Average over all trials

# In[244]:


for i in range(1, ns+1):
    C3_avg = 'C3_0' + str(i) + '_avg'
    C4_avg = 'C4_0' + str(i) + '_avg'

    globals()[C3_avg] = np.mean(C3, axis=0)
    globals()[C4_avg] = np.mean(C4, axis=0)


# In[246]:


C3_01_avg.shape, C4_01_avg.shape


# In[267]:


data = C3_01_avg


plt.figure(figsize=(15, 8))
fs = 250
pl = 20*np.log10(np.abs(np.fft.rfft(data[0])))
pr = 20*np.log10(np.abs(np.fft.rfft(data[1])))
f = np.linspace(0, fs/2, len(pl))

plt.plot(f, pl, c='k')
plt.plot(f, pr, c='y')
plt.grid()
# plt.xticks(np.arange(0, f, 10))
plt.show()


# In[263]:


f.shape


# In[261]:


np.arange(0, f, 10)


# In[ ]:





# In[ ]:





# #### Average over all subjects
# Average C3 and C4 over all subject

# In[ ]:


left_sum = 0
right_sum = 0
for i in range(1, ns+1):
    left = globals()['left0' + str(i) + '_avg']
    right = globals()['right0' + str(i) + '_avg']
    
    left_sum = left_sum + left[0]
    right_sum = right_sum + right[0]


left_avg = left_sum/9
right_avg = right_sum/9


# In[ ]:


left_avg.shape, right_avg.shape


# In[ ]:


# summ = left01_avg[0] + left02_avg[0] + left03_avg[0] + left04_avg[0] + left05_avg[0] + left06_avg[0] + left07_avg[0] + left08_avg[0] + left09_avg[0]


# In[ ]:


side = input('Which side (left/right)? ')

data = globals()[side + '_avg']
figure = plt.figure(figsize = (15, 9))

# ================== C3 =========================
ax1 = figure.add_subplot(221)
ax1.set_title('C3')
ax1.plot(xax, data)
ax1.plot([0.5]*50, range(-7, 43), linewidth = 3, c='r', linestyle='dashed')
ax1.plot([1.5]*50, range(-7, 43), 'g--', linewidth = 3)

ax1.set_xticks(np.arange(0, 5, 0.5))
ax1.axis(ymin=-7,ymax=20)

ax1.annotate('Cue Start', xy=(0.5, 10),  xycoords='data',
            xytext=(1.2, 12),
            arrowprops=dict(facecolor='black', width=1),
            horizontalalignment='right', verticalalignment='top',
            size=12)
ax1.annotate('MI Start', xy=(1.5, 10),  xycoords='data',
            xytext=(2.3, 12),
            arrowprops=dict(facecolor='black', width=1),
            horizontalalignment='right', verticalalignment='top',
            size=12)
ax1.set_xlabel('Second')
ax1.set_ylabel('uV?')

ax2 = figure.add_subplot(223)
p = 20*np.log10(np.abs(np.fft.rfft(data)))
f = np.linspace(0, fs/2, len(p))

ax2.plot(f, p)
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Power')


# =================== C4 ==========================
ax1 = figure.add_subplot(222)
ax1.set_title('C4')
ax1.plot(xax, data)
ax1.plot([0.5]*50, range(-10, 40), linewidth = 3, c='r', linestyle='dashed')
ax1.plot([1.5]*50, range(-10, 40), 'g--', linewidth = 3)

ax1.set_xticks(np.arange(0, 5, 0.5))
ax1.axis(ymin=-10,ymax=20)

ax1.annotate('Cue Start', xy=(0.5, 10),  xycoords='data',
            xytext=(1.2, 12),
            arrowprops=dict(facecolor='black', width=1),
            horizontalalignment='right', verticalalignment='top',
            size=12)
ax1.annotate('MI Start', xy=(1.5, 10),  xycoords='data',
            xytext=(2.3, 12),
            arrowprops=dict(facecolor='black', width=1),
            horizontalalignment='right', verticalalignment='top',
            size=12)
ax1.set_xlabel('Second')
ax1.set_ylabel('uV?')

ax2 = figure.add_subplot(224)
p = 20*np.log10(np.abs(np.fft.rfft(data)))
f = np.linspace(0, fs/2, len(p))

ax2.plot(f, p)
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Power')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


y = left01_avg[1]
y


# 1 

# In[ ]:


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


data = np.random.rand(301) - 0.5


# In[ ]:


ps.shape


# In[ ]:


time_step = 1 / 250
freqs = np.fft.fftfreq(y.size, time_step)


# In[ ]:


freqs.shape


# In[ ]:


idx = np.argsort(freqs)


# In[ ]:


plt.plot(freqs, ps)
plt.xlim([0, 20])


# 2 https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python

# In[ ]:





# In[ ]:


import numpy as np
import pylab as pl

rate = 250.0
t = np.arange(0, 10, 1/rate)
x = np.sin(2*np.pi*4*t) + np.sin(2*np.pi*7*t) + np.random.randn(len(t))*0.2
p = 20*np.log10(np.abs(np.fft.rfft(data[0])))
f = np.linspace(0, rate/2, len(p))
plt.plot(f, p)
plt.show()


# In[ ]:





# 3

# In[ ]:


sampling_rate = 250.0

time = np.arange(0, 10, 1/sampling_rate)

# data = np.sin(2*np.pi*6*time) + np.random.randn(len(time))
data = y

fourier_transform = np.fft.rfft(data)

abs_fourier_transform = np.abs(fourier_transform)

power_spectrum = np.square(abs_fourier_transform)

frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))

plt.plot(frequency, power_spectrum)
plt.xlim([0, 20])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from scipy import signal
import matplotlib.pyplot as plt

f, Pxx_den = signal.welch(y, 250, nperseg=1024)
plt.semilogy(f, Pxx_den)
# plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


# In[ ]:


from scipy.fftpack import fft

yf = fft(y)  

plt.subplot(211)
plt.plot(idx_test, left01_avg[0])

plt.subplot(212)
plt.plot(idx_test, np.abs(yf)**2)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

dt = 0.01
t = np.arange(0, 10, dt)
nse = np.random.randn(len(t))
r = np.exp(-t / 0.05)


# In[ ]:





# In[ ]:


cnse = np.convolve(nse, r) * dt
cnse = cnse[:len(t)]
s = 0.1 * np.sin(2 * np.pi * t) + cnse
plt.subplot(211)
plt.plot(t, s)

plt.subplot(212)
plt.psd(s, 512, 1 / dt)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Confusion matrix

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# In[ ]:


cm = np.array(confusion_matrix(ytest, ypred, labels=[1, 0]))


# In[ ]:


confusion = dframe(cm, index=['left hand', 'right hand'],
                columns=['predicted_left', 'predicted_right'])


# In[ ]:


confusion


# In[ ]:


sns.heatmap(confusion, annot=True)


# In[ ]:


print(classification_report(ytest, ypred))


# In[ ]:




