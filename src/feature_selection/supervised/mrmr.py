#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_selection import f_regression as sklearn_f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

FLOOR = .001


def groupstats2fstat(avg, var, n):
    """Compute F-statistic of some variables across groups
    Compute F-statistic of many variables, with respect to some groups of instances.
    For each group, the input consists of the simple average, variance and count with respect to each variable.
    Parameters
    ----------
    avg: pandas.DataFrame of shape (n_groups, n_variables)
        Simple average of variables within groups. Each row is a group, each column is a variable.
    var: pandas.DataFrame of shape (n_groups, n_variables)
        Variance of variables within groups. Each row is a group, each column is a variable.
    n: pandas.DataFrame of shape (n_groups, n_variables)
        Count of instances for whom variable is not null. Each row is a group, each column is a variable.
    Returns
    -------
    f: pandas.Series of shape (n_variables, )
        F-statistic of each variable, based on group statistics.
    Reference
    ---------
    https://en.wikipedia.org/wiki/F-test
    """
    avg_global = (avg * n).sum() / n.sum()  # global average of each variable
    numerator = (n * ((avg - avg_global) ** 2)).sum() / (len(n) - 1)  # between group variability
    denominator = (var * n).sum() / (n.sum() - len(n))  # within group variability
    f = numerator / denominator
    return f.fillna(0.0)


def mrmr_base(K, relevance_func, redundancy_func,
              relevance_args={}, redundancy_args={},
              denominator_func=np.mean, only_same_domain=False):
    """General function for mRMR algorithm.
    Parameters
    ----------
    K: int
        Maximum number of features to select. The length of the output is *at most* equal to K
    relevance_func: callable
        Function for computing Relevance.
        It must return a pandas.Series containing the relevance (a number between 0 and +Inf)
        for each feature. The index of the Series must consist of feature names.
    redundancy_func: callable
        Function for computing Redundancy.
        It must return a pandas.Series containing the redundancy (a number between -1 and 1,
        but note that negative numbers will be taken in absolute value) of some features (called features)
        with respect to a variable (called target_variable).
        It must have *at least* two parameters: "target_variable" and "features".
        The index of the Series must consist of feature names.
    relevance_args: dict (optional, default={})
        Optional arguments for relevance_func.
    redundancy_args: dict (optional, default={])
        Optional arguments for redundancy_func.
    denominator_func: callable (optional, default=numpy.mean)
        Synthesis function to apply to the denominator of MRMR score.
        It must take an iterable as input and return a scalar.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary redundancy coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    global score_denominator
    relevance = relevance_func(**relevance_args)
    features = relevance[relevance.fillna(0) > 0].index.to_list()
    relevance = relevance.loc[features]
    redundancy = pd.DataFrame(FLOOR, index=features, columns=features)
    K = min(K, len(features))
    selected_features = []
    not_selected_features = features.copy()

    for i in tqdm(range(K)):

        score_numerator = relevance.loc[not_selected_features]

        if i > 0:

            last_selected_feature = selected_features[-1]

            if only_same_domain:
                not_selected_features_sub = [c for c in not_selected_features if
                                             c.split('_')[0] == last_selected_feature.split('_')[0]]
            else:
                not_selected_features_sub = not_selected_features

            if not_selected_features_sub:
                redundancy.loc[not_selected_features_sub, last_selected_feature] = redundancy_func(
                    target_column=last_selected_feature,
                    features=not_selected_features_sub,
                    **redundancy_args
                ).fillna(FLOOR).abs().clip(FLOOR)
                score_denominator = redundancy.loc[not_selected_features, selected_features].apply(
                    denominator_func, axis=1).replace(1.0, float('Inf'))

        else:
            score_denominator = pd.Series(1, index=features)

        score = score_numerator / score_denominator

        best_feature = score.index[score.argmax()]
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)

    return selected_features


def parallel_df(func, df, series):
    n_jobs = min(cpu_count(), len(df.columns))
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(df.iloc[:, col_chunk], series)
        for col_chunk in col_chunks
    )
    return pd.concat(lst)


def _f_classif(X, y):
    def _f_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_classif(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_classif_series(col, y)).fillna(0.0)


def _f_regression(X, y):
    def _f_regression_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_regression(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_regression_series(col, y)).fillna(0.0)


def f_classif(X, y):
    return parallel_df(_f_classif, X, y)


def f_regression(X, y):
    return parallel_df(_f_regression, X, y)


def random_forest_classif(X, y):
    forest = RandomForestClassifier(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def random_forest_regression(X, y):
    forest = RandomForestRegressor(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def correlation(target_column, features, X):
    def _correlation(X, y):
        return X.corrwith(y).fillna(0.0)
    return parallel_df(_correlation, X.loc[:, features], X.loc[:, target_column])


def encode_df(X, y, cat_features, cat_encoding):
    ENCODERS = {
        'leave_one_out': ce.LeaveOneOutEncoder(cols=cat_features, handle_missing='return_nan'),
        'james_stein': ce.JamesSteinEncoder(cols=cat_features, handle_missing='return_nan'),
        'target': ce.TargetEncoder(cols=cat_features, handle_missing='return_nan')
    }
    X = ENCODERS[cat_encoding].fit_transform(X, y)
    return X


def mrmr_classif(
        X, y, K,
        relevance='f', redundancy='c', denominator='mean',
        cat_features=[], cat_encoding='leave_one_out',
        only_same_domain=False
):
    """MRMR feature selection for a classification task
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (categorical) target variable.
    K: int
        Number of features to select.
    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method.
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    relevance_func = f_classif if relevance=='f' else (
                     random_forest_classif if relevance=='rf' else relevance)
    redundancy_func = correlation if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
                       np.max if denominator == 'max' else denominator)

    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}

    selected_features = mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                                  relevance_args=relevance_args, redundancy_args=redundancy_args,
                                  denominator_func=denominator_func, only_same_domain=only_same_domain)

    return selected_features


def mrmr_regression(
        X, y, K,
        relevance='f', redundancy='c', denominator='mean',
        cat_features=[], cat_encoding='leave_one_out',
        only_same_domain=False
):
    """MRMR feature selection for a regression task
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (categorical) target variable.
    K: int
        Number of features to select.
    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method.
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """
    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    relevance_func = f_regression if relevance=='f' else (
                     random_forest_regression if relevance=='rf' else relevance)
    redundancy_func = correlation if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
                       np.max if denominator == 'max' else denominator)

    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}

    selected_features = mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                                  relevance_args=relevance_args, redundancy_args=redundancy_args,
                                  denominator_func=denominator_func, only_same_domain=only_same_domain)

    return selected_features


# In[ ]:




