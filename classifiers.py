from numpy import argmin
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


def classify_all(classifier, df):
    return [classifier.classify(df.iloc[i]) for i in tqdm(range(df.shape[0]))]


class MinimumDistanceClassifier:

    def __init__(self, df):
        groups = df.groupby('ACTIVITY')
        self.classes = []
        self.class_prototypes = []
        for name, group in groups:
            self.classes.append(name)
            self.class_prototypes.append(group.mean().values)

    def classify(self, data):
        distances = euclidean_distances([data], self.class_prototypes)
        min_index = argmin(distances)
        return self.classes[min_index]
