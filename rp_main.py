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
from sklearn.metrics.pairwise import euclidean_distances


# In[2]:


path = join('wisdm-dataset', 'arff_files')
devices = ['phone', 'watch']
sensors = ['accel', 'gyro']

def fix_arff_files():
    for d in devices:
        for s in sensors:
            cur_path = join(path, d, s)
            files = [f for f in listdir(cur_path) if f.endswith('.arff')]
            for f in files:
                data = ''
                with open(join(cur_path, f), 'r+') as file:
                    data = file.read()
                    data = data.replace('{ ', '{')
                    data = data.replace(' }', '}')
                    file.seek(0)
                    file.truncate()
                    file.write(data)
                    file.close()


def arff_to_df(cur_path):
    dt = arff.load(open(cur_path))
    return pd.DataFrame(dt['data'], columns=[row[0] for row in dt['attributes']])


def read_all_files():
    df = {}
    for d in devices:
        for s in sensors:
            print(d, s)
            cur_path = join(path, d, s)
            df[d, s] = pd.concat((arff_to_df(join(cur_path, f)) for f in listdir(cur_path) if f.endswith('.arff')))
    return df


# In[3]:


def translate_activity():
    act_dict = {}
    not_jogg = []
    with open(join('wisdm-dataset','activity_key.txt'),'r' ) as file:
        data = file.read()
        data = data.split('\n')
        for i in range(len(data)-2):
            data[i] = data[i].split(' ')
            act_dict.update({data[i][2]:data[i][0]})
            if data[i][0]!='jogging':
                not_jogg.append(data[i][2])
    
    return not_jogg


# In[4]:


df = read_all_files()


# In[5]:


cur_device = devices[0]
cur_sensor = sensors[0]

def drop_non_features(df, device, sensor):
    return df[cur_device,cur_sensor].drop(['ACTIVITY', 'class'], axis=1)

cur_df = drop_non_features(df, cur_device, cur_sensor)
#print(cur_df)

def normalize_df(df):
    #return df.iloc[:,:] = preprocessing.Normalizer(norm='l2').fit_transform(df)
    return ((df-df.mean())/df.std())
    

cur_df = normalize_df(cur_df)


# In[11]:


def drop_correlated(df, min_corr=0.95):

    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    dropped_features = [col for col in upper.columns if any(upper[col] > min_corr)]
    print(dropped_features)
    # Drop features
    return df.drop(df[dropped_features], axis=1)

cur_df = drop_correlated(cur_df)


# In[8]:


#Calculate class prototypes
not_jogg = translate_activity()
temp = df[cur_device, cur_sensor]
temp['ACTIVITY'] = temp['ACTIVITY'].replace(to_replace= not_jogg,value='A')
activities = temp['ACTIVITY'].unique()
class_means = [cur_df[temp['ACTIVITY']==act].mean().values for act in activities]


# In[9]:


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


# In[32]:


#model = PCA(.99) # Keep n components that fit 99% of data
#model.fit(cur_df)
#print(model.n_components_)

model = PCA(n_components = 2)
pc = model.fit_transform(cur_df)

pca_df = pd.DataFrame(data=pc, columns=["pc1","pc2"])
pca_df.reset_index(drop=True, inplace=True)
temp['ACTIVITY'].reset_index(drop=True, inplace=True)
pca_df = pd.concat([pca_df,temp['ACTIVITY']],axis=1)

fig = plt.figure()
sub = fig.add_subplot(1,1,1)
sub.set_xlabel('Principal Component 1')
sub.set_ylabel('Principal Component 2')
sub.set_title('2 component PCA')

targets = ['A', 'B']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = pca_df['ACTIVITY'] == target
    sub.scatter(pca_df.loc[indicesToKeep, 'pc1']
               , pca_df.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
sub.legend(targets)
sub.grid()

print(model.explained_variance_ratio_)

