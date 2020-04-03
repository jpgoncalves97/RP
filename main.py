#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install liac-arff
import arff
from os.path import join
from os import listdir
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances


# In[23]:


path = join('wisdm-dataset', 'arff_files')
devices = ['phone', 'watch']
sensors = ['accel', 'gyro']


def arff_to_df(cur_path):
    dt = arff.load(open(cur_path))
    print(type(dt))
    return pd.DataFrame(dt['data'], columns=[row[0] for row in dt['attributes']])


def fix_arff_files():
    for d in devices:
        for s in sensors:
            cur_path = join(path, d, s)
            files = [f for f in listdir(cur_path) if f.endswith('.arff')]
            for f in files:
                with open(join(cur_path, f), 'r+') as file:
                    data = file.read()
                    data.replace('{ ', '{')
                    data.replace(' }', '}')
                    file.seek(0)
                    file.truncate()
                    file.write(data)


def read_all_files():
    df = {}
    for d in devices:
        for s in sensors:
            print(d, s)
            cur_path = join(path, d, s)
            df[d, s] = pd.concat((arff_to_df(join(cur_path, f)) for f in listdir(cur_path) if f.endswith('.arff')))
    return df


# In[24]:


df = read_all_files()


# In[26]:


cur_device = devices[0]
cur_sensor = sensors[0]

def drop_non_features(df, device, sensor):
    return df[cur_device,cur_sensor].drop(['ACTIVITY', 'class'], axis=1)

cur_df = drop_non_features(df, cur_device, cur_sensor)

def normalize_df(df):
    return (df-df.mean())/df.std()

cur_df = normalize_df(cur_df)


# In[27]:


def drop_correlated(df, min_corr=0.95):

    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    dropped_features = [col for col in upper.columns if any(upper[col] > min_corr)]
    # Drop features
    return df.drop(df[dropped_features], axis=1)

cur_df = drop_correlated(cur_df)


# In[28]:


#Calculate class prototypes
temp = df[cur_device, cur_sensor]
activities = temp['ACTIVITY'].unique()
class_means = [cur_df[temp['ACTIVITY']==act].mean().values for act in activities]


# In[29]:


n_rows = cur_df.shape[0]
correct_classified = 0
# Minimum distance classifier
for i in range(n_rows):
    cur_sample = [cur_df.iloc[i]]
    min_dist_ind = np.argmin(euclidean_distances(cur_sample, class_means))
    if activities[min_dist_ind] == temp['ACTIVITY'].iloc[i]:
        correct_classified += 1
print('Correct:', correct_classified)
print('Wrong:', n_rows-correct_classified)
print('Percent', correct_classified/n_rows)


# In[6]:


model = PCA(.99) # Keep n components that fit 99% of data
model.fit(cur_df)
print(model.n_components_)


# In[10]:





# In[ ]:




