import numpy as np


def normalize(df):
    act = df["ACTIVITY"]
    num_df = numeric_df(df)
    df = (num_df - num_df.mean()) / num_df.std()
    df["ACTIVITY"] = act
    return df


def drop_correlated_features(df, min_corr=0.95):
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    corr_features = [col for col in upper.columns if any(upper[col] > min_corr)]
    return df.drop(corr_features, axis=1), corr_features


def numeric_df(df):
    return df.select_dtypes(include=np.number)


def matrix_stats(df):
    fp = df.sum(axis=0) - np.diag(df)
    fn = df.sum(axis=1) - np.diag(df)
    tp = np.diag(df)
    tn = df.values.sum() - (fp + fn + tp)

    # Sensitivity, hit rate, recall, or true positive rate
    tpr = tp / (tp + fn)
    # Specificity or true negative rate
    tnr = tn / (tn + fp)

    # Overall accuracy
    acc = (tp + tn) / (tp + fp + fn + tn)
    return acc, tpr, tnr, fp, fn, tp, tn
