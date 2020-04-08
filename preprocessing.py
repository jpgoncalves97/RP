import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from scipy.stats.mstats import kruskalwallis
import pandas as pd


# https://stackoverflow.com/questions/21164910/how-do-i-delete-a-column-that-contains-only-zeros-in-pandas
def drop_zero_columns(df):
    return df.loc[:, (df != 0).any(axis=0)]


def normalize(df):
    numeric_cols = df.dtypes == np.number
    norm_df = df.loc[:, numeric_cols]
    df.loc[:, numeric_cols] = (norm_df - norm_df.mean()) / norm_df.std()
    return df


def drop_correlated_features(df, min_corr=0.95):
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    corr_features = [col for col in upper.columns if any(upper[col] > min_corr)]
    return df.drop(corr_features, axis=1), corr_features


def kruskal(df, alpha=0.05):
    num_df = df.select_dtypes(include=np.number)
    kruskal_pvalues = np.empty(len(num_df.columns))
    for ind, col in enumerate(num_df.columns):
        test = kruskalwallis(*[group[col].values for name, group in df.groupby("ACTIVITY")])
        kruskal_pvalues[ind] = test.pvalue
    return num_df.columns[kruskal_pvalues > alpha].values


def pca(data, components=0.99):
    if type(components) == int:
        model = PCA(n_components=components)
        new_data = model.fit(data).transform(data)
        return new_data, model, model.n_components, model.explained_variance_ratio_
    else:
        model = PCA()
        new_data = model.fit(data).transform(data)
        n_components, explained_ratio = get_components(model.explained_variance_ratio_, components)
        return new_data[:, :n_components], model, n_components, explained_ratio


def lda(data, targets, components=0.99):
    if type(components) == int:
        model = LDA(n_components=components)
        new_data = model.fit(data, targets).transform(data)
        return new_data, model, model.n_components, model.explained_variance_ratio_
    else:
        model = LDA()
        new_data = model.fit(data, targets).transform(data)
        n_components, explained_ratio = get_components(model.explained_variance_ratio_, components)
        return new_data[:, :n_components], model, n_components, explained_ratio


def get_components(explained_ratios, percent):
    total = 0
    for ind, val in enumerate(explained_ratios):
        total += val
        if total >= percent:
            break
    return ind + 1, total


def get_reduced_df(data, old_df):
    new_df = pd.DataFrame(data)
    new_df.insert(0, 'ACTIVITY', old_df['ACTIVITY'].to_numpy())
    new_df.insert(new_df.shape[1], 'TARGET', old_df['TARGET'].to_numpy())
    return new_df
