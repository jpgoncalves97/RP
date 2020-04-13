#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install liac-arff
# pip install tqdm
from sklearn.metrics import confusion_matrix
from copy import deepcopy


from data_import import *
from classifiers import *
from preprocessing import *

import pandas as pd

# In[2]:


path = join('wisdm-dataset', 'arff_files')
devices = ['phone', 'watch']
sensors = ['accel', 'gyro']

act_df = read_activities()

df = read_files(path, 'phone', 'gyro', act_df)

# In[3]:


print('Original number of features', df.shape[1] - 1)
df = df.dropna()
print('After dropping NaN', df.shape[1] - 1)
df = drop_zero_columns(df)
print('After dropping columns with all 0s', df.shape[1] - 1)
df = normalize(df)

scenario = 'A'
df = set_targets(df, act_df, scenario)

# In[4]:


df, corr_features = drop_correlated_features(df)
print('Correlated features:', corr_features)
print('After dropping correlated', df.shape[1] - 2)

# In[5]:


kruskal_features = kruskal(df)
print('Kruskal features:', kruskal_features)
df = df.drop(kruskal_features, axis=1)
print('After dropping kruskal', df.shape[1] - 2)

# In[6]:


new_features, model, n_components, explained_ratio = pca(df.select_dtypes(include=np.number), .99)
# new_features, model, n_components, explained_ratio = lda(df.select_dtypes(include=np.number), df["TARGET"], .99)
print('Old number of components', df.shape[1] - 2)
print('New number of components', n_components)

# In[7]:


new_df = get_reduced_df(new_features, df)
classifier = FisherClassifier(new_df)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)

classifiers = [MinimumDistanceClassifier, FisherClassifier]
scores = [[] for c in classifiers]
tprs = [[] for c in classifiers]
tnrs = [[] for c in classifiers]
for train, test in skf.split(new_df, new_df['TARGET']):
    train_df = new_df.iloc[train]
    test_df = new_df.iloc[test]
    for ind, classifier in enumerate(classifiers):
        c = classifier(train_df)
        labels = np.unique(test_df['TARGET'])
        predictions = classify_all(c, test_df)
        results = get_results(test_df['TARGET'], predictions, labels)
        acc, tpr, tnr = matrix_stats(results)

        scores[ind].append(score_func(tpr, tnr))
        print(tpr[0], tpr[1], sep='\t')
        pass

'''
labels = np.unique(df["TARGET"])
predictions = classify_all(classifier, new_df)
results = get_results(df['TARGET'], predictions, labels)


# In[9]:


plt.figure()
sn.heatmap(results, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()


# In[10]:


acc, tpr, tnr = matrix_stats(results)
result_df = pd.concat((tpr*100, tnr*100, acc*100), axis=1)
result_df.columns = 'tpr', 'tnr', 'acc'
print(result_df)
result_df.plot(kind='bar').set_ylabel('Percentage')
plt.show()
'''

# In[ ]:




