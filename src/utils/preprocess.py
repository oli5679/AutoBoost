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


class TreeBinner(BaseEstimator, TransformerMixin):
    """
    Autobinning tool - automated feature binning for regression models by fitting single-variable trees
    
    Attributes:
        fit (method): finds bin-thresholds for columns X given target y 
        transform (method): bins X according to found bin-thresholds
        fit_transform (method): fit & transform

    """

    def __init__(
        self,
        tree_params=DEFAULT_TREEBINNER_PARAMS,
        random_state=0,
        auc_cutoff=0.55,
        verbose=True,
        suffix="binned",
    ):
        """
        Args:
            tree_params (dict):  configuration parameters, for binning tree
            random_state (int): random seed, default 0
            auc_cutoff (numeric): trees with AUC below this will be ignored, default 0.55
            suffix (string): column suffic
        """
        self.tree_params = tree_params
        self.random_state = random_state
        self.auc_cutoff = auc_cutoff
        self.verbose = verbose
        self.suffix = suffix

        self.model_binning = DecisionTreeClassifier(
            **tree_params, random_state=random_state
        )
        self.splits = defaultdict(lambda: [-np.inf, np.inf])

    def _is_leaf(self, node_id):
        """
        Is particular node in self.model_binning.tree_ a leaf (terminal node)?
        Args:
            node_id: (int)
        Returns:
            True if leaf, False otherwise
        """
        return (
            self.model_binning.tree_.children_left[node_id]
            == self.model_binning.tree_.children_right[node_id]
        )

    def fit(self, X, y):
        """
        Find splits for all cols in X which exceed ROC-AUC threshold

        Args:
            X (dataframe): dataframe of numeric columns to be binned
            y (series): binary target variable used for binning

        Returns 
            self (object): 

        """
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        for col in X.columns:
            self._get_splits(col)

        return self

    def _get_splits(self, feature):
        """
        Find splits for single feature if exceeding ROC-AUC threshold

        Args:
            feature (str)
        """
        X_train_feature = self.X_train_.loc[:, feature].values.reshape(-1, 1)
        X_test_feature = self.X_test_.loc[:, feature].values.reshape(-1, 1)

        self.model_binning.fit(X_train_feature, self.y_train_)

        y_pred = self.model_binning.predict_proba(X_test_feature)[:, 1]

        auc_score = roc_auc_score(self.y_test_, y_pred)
        if (auc_score > self.auc_cutoff) & (
            len(self.X_train_.loc[:, feature].unique()) > 2
        ):
            self._unpack_splits(feature)

            if self.verbose:
                logger.info(
                    "Binning feature {} with auc on test: {:.4}".format(
                        feature, auc_score
                    )
                )
                logger.info(f"\tBins: {feature} - {self.splits[feature]}")

    def _unpack_splits(self, feature):
        """
        Unpacks splits for single feature from self.model_binning.tree_, and saves to self.splits

        Args:
            feature (str)
        """
        stack = [0]
        while len(stack) > 0:
            node_id = stack.pop()
            if not self._is_leaf(node_id):
                left_child = self.model_binning.tree_.children_left[node_id]
                right_child = self.model_binning.tree_.children_right[node_id]
                stack.append(left_child)
                stack.append(right_child)
                if self._is_leaf(left_child) or self._is_leaf(right_child):
                    self.splits[feature].append(
                        self.model_binning.tree_.threshold[node_id]
                    )
        self.splits[feature] = sorted(self.splits[feature])

    def transform(self, X, y=None):
        """
        Returns data binned by discovered splits

        Args:
            X (dataframe): data to be binned

        Returns:
            X_binned (dataframe): binned dataframe
        """
        for feature in self.splits.keys():
            if self.splits[feature] != [-np.inf, np.inf]:
                X[feature + self.suffix] = pd.cut(
                    X[feature], self.splits[feature]
                ).astype(str)
        return X
