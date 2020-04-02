#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arff
from os.path import join
from os import listdir
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


path = join('wisdm-dataset', 'arff_files')
devices = ['phone', 'watch']
sensors = ['accel', 'gyro']


def arff_to_df(cur_path):
    dt = arff.load(open(cur_path))
    return pd.DataFrame(dt['data'], columns=[row[0] for row in dt['attributes']])


def fix_arff_files():
    for d in devices:
        for s in sensors:
            cur_path = join(path, d, s)
            files = [f for f in listdir(cur_path) if f.endswith('.arff')]
            for f in files:
                data = ''
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


# In[3]:


df = read_all_files()


# In[4]:


def standardize_df(df):
    return (df-df.mean())/df.std()

cur_df = standardize_df(df['phone','accel'].iloc[:,1:-1])


# In[5]:


def drop_correlated(df, min_corr=0.95):

    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    dropped_features = [col for col in upper.columns if any(upper[col] > min_corr)]
    # Drop features
    return df.drop(df[dropped_features], axis=1)

cur_df = drop_correlated(cur_df)


# In[6]:


model = PCA(.99) # Keep n components that fit 99% of data
model.fit(cur_df)
print(model.n_components_)


# In[ ]:




