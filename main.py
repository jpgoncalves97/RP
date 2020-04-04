#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install liac-arff
#pip install tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

from data_import import *
from classifiers import *
from preprocessing import *

import pandas as pd


# In[2]:


path = join('wisdm-dataset', 'arff_files')
devices = ['phone', 'watch']
sensors = ['accel', 'gyro']

activities = read_activities()

df = read_files(path, 'phone', 'accel')
targets = df["ACTIVITY"]


# In[3]:


df = normalize(df)


# In[4]:


df, corr_features = drop_correlated_features(df)


# In[5]:


classifier = MinimumDistanceClassifier(df)
num_df = numeric_df(df)
predictions = classify_all(classifier, num_df)
results = pd.DataFrame(confusion_matrix(targets, predictions))#, index=activities["code"], columns=activities["code"])


# In[6]:


plt.figure(figsize = (15, 15))
labels=activities["code"]
sn.heatmap(results, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()


# In[6]:


acc, tpr, tnr, fp, fn, tp, tn  = matrix_stats(results)
print(acc)
print(tpr)
print(tnr)


# In[19]:





# In[ ]:




