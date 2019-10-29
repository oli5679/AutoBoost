import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_TREEBINNER_PARAMS = {"criterion": "gini", "min_samples_leaf": 0.05}


class AutoEncoder(BaseEstimator, TransformerMixin):
    """
    Automatically encodes features in a way that plays nicely with GBDT
        - dates --> hour + day of year + numeric
        - categoricals with ordinal ranking --> ordinal ranking
        - high cardiality categoricals --> frequency counts & string lenghts
        - low cardiality categoricals --> one-hot-encode & fill in missing columns from transform phase

    Attributes:
        fit (method): finds preprocessing strategies for all object/date cols
        transform (method): preprocesses all object/date cols
        fit_transform (method): fit & transform

    NOTE - one-hot-encoding for compatability with SHAP library, this might cause memory issues for large datasets
    """

    def __init__(self, ordinality_mapping={}, cardinality_threshold=100, date_cols=[]):
        """
        Args:
            cardinality_threshold (numeric): number of unique values to count-encode, rather than one-hot-encode
            date_cols (list): date-type columns
        """
        self.cardinality_threshold = cardinality_threshold
        self.date_cols = date_cols
        self.ordinality_mapping = ordinality_mapping

    def _one_hot_encode(self, X, col):
        """
        Args:
            X (dataframe): data to be one-hot-encoded
            col (string): column to be one-hot-encoded

        Returns:
            X (dataframe): data, one-hot-encoded, including missing categories found in 'fit' method
        """
        X = pd.get_dummies(X, columns=[col])
        # add in missing cols
        for c in self.ohe_encode_cols_[col]:
            if c not in X.columns:
                X[c] = 0
        return X

    def fit(self, X, y=None):
        """
        Args:
            X (dataframe): data to be encoded

        Returns:
            self (BaseEstimator): fitted transformer
        """
        self.count_map_cols_ = {}
        self.ohe_encode_cols_ = {}
        self.cat_cols_ = [
            c
            for c in X.loc[:, X.dtypes == "object"].columns
            if c not in self.ordinality_mapping.keys()
        ]
        for col in self.cat_cols_:
            uniq_vals = list(X[col].unique())
            if len(uniq_vals) >= self.cardinality_threshold:
                self.count_map_cols_[col] = Counter(X[col])
            else:
                self.ohe_encode_cols_[col] = [col + "_" + v for v in uniq_vals]
        return self

    def transform(self, X, y=None):
        """
        Args:
            X (dataframe): data to be encoded

        Returns:
            X (dataframe): encoded data
        """
        for col, mapping in self.ordinality_mapping.items():
            X[col] = X[col].map(mapping)

        for col, mapping in self.count_map_cols_.items():
            X[f"len_{col}"] = X[col].str.len()
            X[col] = X[col].map(mapping)

        for col in self.ohe_encode_cols_.keys():
            X = self._one_hot_encode(X, col)

        for col in self.date_cols:
            X[col + "_hour"] = X[col].dt.hour + (X[col].dt.minute / 60)
            X[col + "_month"] = X[col].dt.month + (X[col].dt.day / 30)
        X[self.date_cols] = X[self.date_cols].astype(int)
        return X


class ColumnRemover(BaseEstimator, TransformerMixin):
    """
    Drops specified colums from Dataframe

    Attributes:
        fit (method): finds columns to be dropped
        transform (method): drops columns
        fit_transform (method): fit & transform

    """

    def __init__(self, drop_columns):
        """
        Args:
            drop_columns (list): columns to be dropped
        """
        self.drop_columns = drop_columns

    def fit(self, X, y=None):
        """
        Args:
            X (dataframe): data to have columns dropped

        Returns:
            self (BaseEstimator): fitted transformer
        """
        self.columns = [c for c in X.columns if c not in self.drop_columns]
        return self

    def transform(self, X):
        """
        Args:
            X (dataframe): data to drop columns

        Returns:
            X (dataframe): data with dropped columns
        """
        return X[self.columns]
