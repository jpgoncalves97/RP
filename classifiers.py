import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sn


def get_conf_matrix(targets, predictions):
    labels = np.unique(targets)
    return pd.DataFrame(confusion_matrix(targets, predictions), columns=labels, index=labels)


def plot_conf_matrix(targets, predictions):
    labels = np.unique(targets)
    matrix = get_conf_matrix(targets, predictions)
    plt.figure()
    sn.heatmap(matrix, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()


def set_targets(df, act_df, scenario):
    if scenario == 'A':
        df['TARGET'] = df.apply(lambda row: 'jogging' if row['ACTIVITY'] == 'B' else 'not jogging', axis=1)
    elif scenario == 'B':
        df['TARGET'] = df.apply(lambda row: act_df.loc[row['ACTIVITY'], 'group'], axis=1)
    elif scenario == 'C':
        df['TARGET'] = df.apply(lambda row: act_df.loc[row['ACTIVITY'], 'desc'], axis=1)
    return df

# https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal

def classify_all(classifier, df):
    df = df.select_dtypes(include=np.number)
    return [classifier.fit(df.iloc[i]) for i in tqdm(range(df.shape[0]))]


def get_results(targets, predictions):
    conf_matrix = get_conf_matrix(targets, predictions)

    tp = conf_matrix.loc['jogging', 'jogging']
    fn = conf_matrix.loc['jogging', 'not jogging']
    tn = conf_matrix.loc['not jogging', 'not jogging']
    fp = conf_matrix.loc['not jogging', 'jogging']

    acc = (tp+tn)/(tp+tn+fn+fp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    return acc, tpr, tnr, tp, tn, fp, fn


def cross_validation(df, n_splits, classifiers):
    skf = StratifiedKFold(n_splits=n_splits)
    accs = [[] for _ in range(len(classifiers))]
    tprs = [[] for _ in range(len(classifiers))]
    tnrs = [[] for _ in range(len(classifiers))]
    targets = []
    predictions = [[] for _ in range(len(classifiers))]
    print('Cross validation')
    for i, (train, test) in enumerate(skf.split(df, df['TARGET'])):
        print(i+1)
        train_df = df.iloc[train]
        test_df = df.iloc[test]
        for t in test_df['TARGET']:
            targets.append(t)
        for ind, classifier in enumerate(classifiers):
            c = classifier(train_df)
            cur_predictions = classify_all(c, test_df)
            for p in cur_predictions:
                predictions[ind].append(p)
            se, sp, acc, tp, tn, fp, fn = get_results(test_df['TARGET'], cur_predictions)
            accs[ind].append(acc)
            tprs[ind].append(se)
            tnrs[ind].append(sp)
    n = 4
    for i, c in enumerate(classifiers):
        print(c.__name__)
        print('Accuracy =', round(np.mean(accs[i]), n), '+-', round(np.std(accs[i])*2, n))
        print('Sensitivity =', round(np.mean(tprs[i]), n), '+-', round(np.std(tprs[i]) * 2, n))
        print('Specificity =', round(np.mean(tnrs[i]), n), '+-', round(np.std(tnrs[i]) * 2, n))
    return targets, predictions


def mul_lists(l1, l2):
    val = 0
    for el1, el2 in zip(l1, l2):
        val += el1 * el2
    return val


class MinimumDistanceClassifier:

    def __init__(self, df):
        groups = df.groupby('TARGET')
        self.classes = []
        self.class_prototypes = []
        for name, group in groups:
            self.classes.append(name)
            self.class_prototypes.append(group.mean(numeric_only=True).to_numpy())

    def fit(self, data):
        distances = euclidean_distances([data], self.class_prototypes)
        min_index = np.argmin(distances)
        return self.classes[min_index]


class FisherClassifier:

    '''
    # Para verificar se funciona comparado com os slides
    arr = np.transpose(np.array(
        [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2], [4, 5, 2, 3, 4, 9, 8, 6, 10, 8], [1, 3, 4, 6, 4, 5, 7, 8, 8, 9],
         [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]]))
    df = pd.DataFrame(arr, columns=['TARGET', 'X0', 'X1', 'ACTIVITY'])
    '''

    def __init__(self, df):
        groups = df.groupby('TARGET')
        # -2 porque a df contem target e activity que nao sao features
        self.n_feat = df.shape[1] - 2
        self.classes = []
        self.class_means = []
        for name, group in groups:
            self.classes.append(name)
            self.class_means.append(group.mean(numeric_only=True).to_numpy())
            # Descomentar para testar exemplo em cima
            # self.class_means.append(group.mean().to_numpy()[1:-1])
        self.class_means = np.array(self.class_means)
        self.S1 = np.zeros((self.n_feat, self.n_feat))
        self.S2 = np.zeros((self.n_feat, self.n_feat))
        m1 = self.class_means[0, :]
        m2 = self.class_means[1, :]
        # print('m1', m1, sep='\n')
        # print('m2', m2, sep='\n')
        # [1:-1] = index das features
        # row, m1, m2 são todos 1D (n_feat), converter para 2D
        # np.newaxis = (n_feat) -> (n_feat, 1)
        # reshape(1, -1) = (n_feat) -> (1, n_feat)
        for row in groups.get_group(self.classes[0]).to_numpy():
            row = row[1:-1]
            self.S1 = self.S1 + (np.matmul((row - m1)[:, np.newaxis], (row - m2).reshape(1, -1)))
        for row in groups.get_group(self.classes[1]).to_numpy():
            row = row[1:-1]
            self.S2 = self.S2 + (np.matmul((row - m1)[:, np.newaxis], (row - m2).reshape(1, -1)))
        self.SW = (self.S1 + self.S2).astype(float)
        # print('sw', self.SW, sep='\n')
        self.w = mul_lists(np.linalg.pinv(self.SW), m1 - m2)
        self.w_t = np.transpose(self.w)
        # print('w', self.w, sep='\n')
        self.bias = (mul_lists(self.w_t, m1) + mul_lists(self.w_t, m2)) / 2
        # print('bias', self.bias, sep='\n')

    def fit(self, data):
        return self.classes[0] if mul_lists(self.w_t, data) - self.bias >= 0 else self.classes[1]
