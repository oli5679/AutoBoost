import sklearn
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import subprocess
import pickle
import logging
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold


logger = logging.getLogger(__name__)

LGB_SEARCH_SPACE = {
    "learning_rate": (0.003, 1.0, "log-uniform"),
    "num_leaves": (1, 100),
    "max_depth": (1, 50),
    "min_child_samples": (100, 400),
    "max_bin": (100, 2000),
    "subsample": (0.6, 1.0, "uniform"),
    "min_child_weight": (1, 10),
    "reg_lambda": (1e-3, 1000, "log-uniform"),
    "reg_alpha": (1e-3, 1.0, "log-uniform"),
    "n_estimators": (50, 1600),
}


def evaluate_model(y_actual, y_pred, y_pred_bm, output_dir, model_name="Model"):
    """Compares model predicitons for binary outcome to benchmark model

    Args:
        y_actual (series): outcomes - binary
        y_pred (series): predictions - 0-1 probability
        y_pred_bm (dictionary): predictions - 0-1 probability of benchmark
        output_dir (string): path to save outputs
        model_name (string): name of model   
    """
    model_brier = sklearn.metrics.brier_score_loss(y_actual, y_pred)
    bm_brier = sklearn.metrics.brier_score_loss(y_actual, y_pred_bm)
    model_roc = sklearn.metrics.roc_auc_score(y_actual, y_pred)
    bm_auc = sklearn.metrics.roc_auc_score(y_actual, y_pred_bm)
    title = f"""{model_name} brier {model_brier:.5f} vs. {bm_brier:.5f} benchmark
        \{model_name} ROC-AUC {model_roc:.3f} vs. {bm_auc:.3f} benchmark"""
    logger.info(title)
    plot_roc_auc(y_pred=y_pred, y_true=y_actual, title=title, output_dir=output_dir)


def create_shap_plots(model, X, output_dir, N=10, frac=0.05):
    """Creates SHAP plots showing average marginal impact of features prediction of model:

    Args:
        model (sklearn-type model): binary classifier to be interpreted
        X (dataframe): features to be explained
        output_dir (string): path to save outputs
        N (int): number of partial dependency plots to make
        frac (float): % of X to sample - this can be slow on large datasets

    Returns:
        importance (array): avg. abs. Shap value, higher = more important feature
    """

    # Todo, add some support for categorical cols

    X_sample = X.sample(frac=frac)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # overall summary plot
    shap.summary_plot(shap_values[1], X_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/summary_plot.png")
    plt.close()

    #  rank features by importance - average abs. shap value
    importance = np.abs(shap_values[1]).mean(axis=0)
    features = sorted(list(zip(X_sample.columns, importance)), key=lambda x: -x[1])

    # show partial dependency for top N
    for f in features[:N]:
        ax = shap.dependence_plot(f[0], shap_values[1], X_sample, show=False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/dependency {f}.png")
        plt.close()

    return importance


def find_features_to_remove(importance, X, importance_cutoff=0.05, corr_threshold=0.6):
    """
    Returns features to be removed from X because:
        Importance less than importance_cutoff
        Or correlation > corr_threshold

    Args:
        importance (series): measure of feature importance
        X (df): training set features
        importance_cutoff (numeric): features with importance below this val will be stripped out
        corr_threshold (numeric): maximum allowed correlation with more imortant feature

    Returns:
        remove_targets (list): list of features to remove
    """
    low_importance_cols = list(X.loc[:, importance <= importance_cutoff].columns)

    high_corr_cols = find_high_corr_cols(
        X=X, corr_threshold=corr_threshold, importance=importance
    )

    return list(set(high_corr_cols + low_importance_cols))


def find_high_corr_cols(X, corr_threshold, importance=None):
    """
    Finds columns to be removed due to correlation > threshold

    NOTE, will drop earlier columns first, so sort by increasing importance to keep more important cols

    Args:
         X (df): training set features
         corr_threshold (numeric): maximum allowed correlation with more imortant feature
         importance (series): feature importance rankings, default is reverse of column-order

    Returns:
        high_corr_cols (list): All cols with corr > threshold for a more 'important' feature 
    """
    if importance is None:
        importance = range(X.shape[1])

    importance_ranking = importance.argsort()
    X = X.iloc[:, importance_ranking]
    importance = importance[importance_ranking]

    corr_matrix = X.corr()
    high_corr_cols = []
    n_cols = len(corr_matrix.columns)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            val = corr_matrix.iloc[j, i]
            col = corr_matrix.columns[i]
            row = corr_matrix.index[j]
            if abs(val) >= corr_threshold:
                # logger.infos the correlated feature set and the corr val
                logger.info(
                    f"dropping {col} because of correlation {val:.3f} with with {row}"
                )
                high_corr_cols.append(col)
    return high_corr_cols


def plot_roc_auc(y_pred, y_true, output_dir, title="ROC-AUC curve"):
    """
    Creates ROC-AUC chart, and saves to output_tir

    Args:
        y_pred (series): probability predictions (between 0-1)
        y true (series): actual outcomes (0 or 1)
        output_dir (string): path to save outputs
        title (string): tile of chart
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, label=f"auc={round(auc,4)}")
    plt.title(title)
    plt.legend(loc=4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/model performance.png")
    plt.close()


def dataset_eda(data, output_dir):
    """
    Saves plots with basic overview of dataset to output_dir

    Args:
        data (df): data to be analysed
        output_dir (string): path to save outputs
    """
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        logging.info(data.head())
        data.head().to_csv(f"{output_dir}/plots/data_sample.csv")
        logging.info(data.describe().T)
        data.describe().to_csv(f"{output_dir}/plots/data_description.csv")
        logging.info("\n Null %")
        logging.info(data.isna().mean() * 100)
        logging.info("\n datatypes")
        logging.info(data.dtypes)
    logging.info("\n creating cat-distribution charts")
    for cat_var in data.loc[:, data.dtypes == "object"].columns:
        if len(data[cat_var].unique()) < 200:
            counts = data[cat_var].value_counts()
            ax = counts.plot.barh()
            plt.title(f"{cat_var} value counts")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/plots/{cat_var} value counts.png")
        else:
            logging.info(f"{cat_var} - {len(data[cat_var].unique())} different vals")

    logging.info("\n pairplot")
    sample_size = min(10000, data.shape[1])
    plot_sample = data.sample(frac=1).head(sample_size)
    ax = sns.pairplot(plot_sample)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/pairplot.png")
    plt.close()

    logging.info("\n correlation plot")
    ax = sns.heatmap(data.corr(), annot=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/corr plot.png")
    plt.close()


def bayes_hyperparam_tune(X, y, search_space, n_iters=20):
    """
    Bayesian tuning for a Lightgbm classifier, efficiently tuning hyperparams
    
    Args:
        X (dataframe): model_features
        y (series): binary target
        search_space (dictionary): parameter space to search
        n_iters (int): number of points in search space to evaluate
        
    Returns:
        results (object): recording of avg. crossvalidated bier-score at each point in search-space

    Todo, refactor to generalise - e.g. use in regression/other model types
    """

    def update_model_status(optim_result):
        """Status callback durring bayesian hyperparameter search"""
        logger.info(
            f"""Model #{len(pd.DataFrame(bayes_cv_tuner.cv_results_))  }
        Best Brier: { np.round(bayes_cv_tuner.best_score_, 7)}
        Best params: { bayes_cv_tuner.best_params_}
        """
        )

    bayes_cv_tuner = BayesSearchCV(
        estimator=lgb.LGBMClassifier(
            objective="binary", metric="auc", n_jobs=-1, verbose=1
        ),
        search_spaces=search_space,
        scoring="brier_score_loss",
        cv=StratifiedKFold(n_splits=5),
        n_jobs=-1,
        n_iter=n_iters,
        verbose=0,
        refit=True,
        error_score=-100,
    )

    return bayes_cv_tuner.fit(X.values, y.values, callback=update_model_status)
