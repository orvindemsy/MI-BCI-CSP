#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_cell_magic('html', '', '<style>\n.code_cell .run_this_cell {\n    display: none;\n}\n</style>')


# After meeting on Tuesday 12 May 2020:

# This code v6 is an attempt in applying Filter Bank Common Spatial Pattern (FBCSP)  
# References:  
# * Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b
# 	doi: 10.3389/fnins.2012.00039
# * Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface
#     doi: 10.1109/IJCNN.2008.4634130

# In[2]:


get_ipython().run_line_magic('autosave', '5')


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

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import pandas as pd
from pandas import DataFrame as dframe


# In[3]:


np.seterr(divide='ignore', invalid='ignore')


# In[4]:


# np.set_printoptions(precision=30, suppress=True)


# In[4]:


# number of subject
ns = 9


# In[5]:


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


# In[6]:


type(data01)


# In[7]:


# Keys available inside data are
data01.files


# In[9]:


data09['artifacts'].shape # Containing artifacts occurent in each event in all runs of a subject


# In[11]:


dframe(data09['artifacts'])


# In[8]:


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


# In[9]:


prop01.shape


# In[12]:


pd.__version__


# In[13]:


pd.options.display.max_rows = None


# In[14]:


dframe(prop02, columns=['etype', 'epos','edur'])


# In[10]:


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


# In[11]:


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


# In[12]:


sample_data01.shape


# In[13]:


# Showing inital value of sample_data
sample_data01.head()


# In[14]:


# Sample data has shape of 67k~ samples
# This samples belong to total of all 9 runs with 48 trials of 4 different classes
for i in range(1, 10):
    var = 'sample_data0' + str(i)
    print(var, globals()[var].shape)


# ## Remove Artifact
# Inside data of each subject exists a 'artifacts' data which contains the information on each trial whether the data is clean, denoted by `0` or containing artifact, denoted by `1`

# In[40]:


# # Append the artifacts column to all events data, shape 288 x 4
# new_prop = np.hstack([all_ev, data['artifacts']])


# In[ ]:


# def removeArtifacts(prop, data):
#     # From prop take all data corresponds to event 769, 770, 771, 772
#     idx = np.asarray(np.where((prop[:, 0]>768) & (prop[:, 0]<773))).flatten()
#     all_ev = prop[idx, :]

#     # Append the artifacts column to all events data, shape 288 x 4
#     new_prop = np.hstack([all_ev, data['artifacts']])

#     # Find index of clean data from all_ev with classes 769 (left) or 770 (right) and artifacts equal to '1'
#     idx_cl = np.argwhere(((new_prop[:, 0]==769) | (new_prop[:, 0]==770)) & (new_prop[:, 3]==0)).flatten()
#     clean_ev = new_prop[idx_cl, :]

#     return clean_ev


# In[ ]:


# # Remove artifacts and take only event 769 (left) and 770 (right)
# clean_ev01 = removeArtifacts(prop01, data01)
# clean_ev02 = removeArtifacts(prop02, data02)
# clean_ev03 = removeArtifacts(prop03, data03)
# clean_ev04 = removeArtifacts(prop04, data04)
# clean_ev05 = removeArtifacts(prop05, data05)
# clean_ev06 = removeArtifacts(prop06, data06)
# clean_ev07 = removeArtifacts(prop07, data07)
# clean_ev08 = removeArtifacts(prop08, data08)
# clean_ev09 = removeArtifacts(prop09, data09)


# In[ ]:


# dframe(clean_ev07)


# In[ ]:


# # Fetch indexes whose 1st column are 769 (left) and 770 (right)
# # Subject 1 - 9
# for i in range(1, 10):
#     idx_l = 'idx0' + str(i) + '_l'
#     idx_r = 'idx0' + str(i) + '_r'
#     clean_ev = globals()['clean_ev0' + str(i)]
#     globals()[idx_l] = np.argwhere(clean_ev[:, 0]==769).flatten()
#     globals()[idx_r] = np.argwhere(clean_ev[:, 0]==770).flatten()


# In[ ]:


idx01_r


# In[ ]:


# # Fetch positions of corresponding event types
# # Subject 01-09
# for i in range(1, 10):
#     pos_l = 'pos0' + str(i) + '_l'
#     pos_r = 'pos0' + str(i) + '_r'
#     clean_ev = globals()['clean_ev0' + str(i)]
#     idx_l = globals()['idx0' + str(i) + '_l']
#     idx_r = globals()['idx0' + str(i) + '_r']

#     globals()[pos_l] = clean_ev[idx_l, 1]
#     globals()[pos_r] = clean_ev[idx_r, 1]


# In[ ]:


# # Duration of both events lasted for 313 samples, we can defined dur as
# cue_dur = 313


# In[ ]:


# idx02_l.shape


# In[ ]:





# ## Without artifacts removal

# According to the documentation we are interested to grab event type of 769, left class and 770, right class
# 
# With each type has duration of 313 samples

# In[15]:


# Fetch indexes whose 1st column are 769 (left) and 770 (right)
# Subject 1 - 9
for i in range(1, 10):
    idx_l = 'idx0' + str(i) + '_l'
    idx_r = 'idx0' + str(i) + '_r'
    prop = globals()['prop0' + str(i)]
    globals()[idx_l] = np.argwhere(prop[:, 0]==769).flatten()
    globals()[idx_r] = np.argwhere(prop[:, 0]==770).flatten()


# In[16]:


idx09_l.shape, idx09_r.shape


# In[17]:


dframe(idx05_l).tail()


# In[18]:


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


# In[19]:


pos09_l.shape, pos09_r.shape


# In[20]:


dframe(pos04_r)


# In[21]:


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


# In[22]:


dur01_l


# In[23]:


# Duration of both events lasted for 313 samples, we can defined dur as
cue_dur = 313


# Be aware that this duration is actually the duration of the cue, let's take a look at the event type table in datasheet
# ![event-type-table.png](./img/event-type-table.png)
# 
# The timing scheme of the paradigm suggest that the MI task last for about 2.75 since after the cue ends
# ![timing-scheme-paradigm.png](./img/timing-scheme-paradigm.png)

# In[24]:


# The amount of sample to be clipped for 1 trial of left and right
sec = 2.75
fs = 250
mi_dur = round(sec * fs)
print('After the 313 samples cue, clip this amount of samples:', mi_dur)


# ## Fetch 688 samples of each event from sample_data, 72 trials each class

# In[25]:


len(pos02_l)


# In[26]:


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


# In[27]:


E04_left.shape, E04_right.shape


# In[28]:


pos04_r


# In[29]:


dframe(prop05)


# In[30]:


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


# In[31]:


# Creating columns for sample data
Xcol = []
for i in range(1, 23):
    if i < 11:
        Xcol.append('EEG0'+str(i))
    else:
        Xcol.append('EEG'+str(i))


# In[32]:


E01_left.shape


# In[33]:


dframe(E01_left[60], columns=Xcol)


# In[34]:


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


# In[35]:


E02_left.shape, E02_right.shape


# In[36]:


dframe(E09_left[50]).head()


# In[ ]:





# ## Stage 1: Bandpass Filter using Chebyshev Type II filter
# Band pass filter will be applied to the EEG data  
# A total of 9 band-pass filters are used:
# 1. 4-8Hz
# 2. 8-12Hz
# 3. 12-16Hz
# 4. 16-20 Hz ...  9. 36-40 Hz
# 
# These band-pass ranges are used because they yield a stable frequency response and cover range of 4-40Hz

# In[72]:


from scipy import signal


# In[156]:


# Example of band-pass filtering with 4-8 frequency band
t = np.linspace(0, len(data), 688, False)
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
# fig.suptitle('XX')

ax[0].plot(t, data)
ax[0].set_title('Original')
sos = signal.cheby2(12, 20, [4, 8], btype='bp', fs=250, output='sos')
filt = signal.sosfilt(sos, data)

ax[1].plot(t, filt, c='k')
ax[0].set_title('Band Pass Filtered')

plt.show()


# In[154]:


filtered.shape


# In[179]:


# Now filter E_left and E_right to 9 portion of frequency bands
for i in range(1, ns+1):
    
    for j in range(E01_left.shape[0]):
    
        startb = 4
        stopb = 8
        inc = 4
        while(stopb < 41):
    
            E_left = globals()['E0' + str(i) + '_left']
            E_right = globals()['E0' + str(i) + '_right']

            filt_left = 'filt0' + str(i) + '_' + str(startb) + '_' + str(stopb) + '_left'
            filt_right = 'filt0' + str(i) + '_' + str(startb) + '_' + str(stopb) + '_right'

            sos_left = signal.cheby2(12, 20, [startb, stopb], btype='bp', fs=250, output='sos')
            sos_right = signal.cheby2(12, 20, [startb, stopb], btype='bp', fs=250, output='sos')

            globals()[filt_left] = signal.sosfilt(sos_left, E_left[j])
            globals()[filt_right] = signal.sosfilt(sos_right, E_right[j])

            startb += inc
            stopb += inc


# In[182]:


dframe(filt01_8_12_left[0])


# In[183]:


dframe(filt01_4_8_left[0])


# In[186]:


plt.figure(figsize=(10, 8))
plt.plot(filt01_4_8_left[0])
plt.plot(filt01_8_12_left[0])
plt.show()


# In[ ]:





# In[150]:


# t = np.linspace(0, 1, 1000, False)
# sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.sin(2*np.pi*16*t) + + np.sin(2*np.pi*18*t)
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(t, sig)
# ax1.set_title('10 Hz and 20 Hz sinusoids')
# ax1.axis([0, 1, -3, 3])

# sos1 = signal.cheby2(0, 20, 16, 'lp', fs=1000, output='sos')
# sos2 = signal.cheby2(7, 20, 18, 'lp', fs=1000, output='sos')

# filtered1 = signal.sosfilt(sos1, sig)
# filtered2 = signal.sosfilt(sos2, sig)

# ax2.plot(t, filtered1)
# ax2.plot(t, filtered2)
# ax2.set_title('After 17 Hz high-pass filter')
# ax2.axis([0, 1, -2, 2])
# ax2.set_xlabel('Time [seconds]')
# plt.show()


# In[103]:


dframe(filtered1)


# In[ ]:




