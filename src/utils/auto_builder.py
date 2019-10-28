import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import subprocess
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from nyoka import lgb_to_pmml
import shutil
import os
from skopt import BayesSearchCV


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
pd.options.mode.chained_assignment = None  # default='warn'

from utils import general, preprocess

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


class AutoBuilder:
    """'
    E2E classifier builder
    
    Builds lightgbm binary classifier, including:
        - dataset EDA (optional)
        - feature preprocessing
        - hyperparameter tuning (optional)
        - model performance assessment
        - SHAP-based feature analysis
        - feature selection
        - creating deployment package (pmml & pkl)

    Attributes:
        auto_build (method): automatically populates output_dir path, with model artifacts, and evaluation charts
    """

    def __init__(
        self,
        output_dir_path,
        csv_path,
        target_col="target",
        target_mapping={0: 0, 1: 1},
        drop_cols=[],
        date_cols=[],
        ord_cols_mapping={},
        eda_flag=True,
        tune_flag=True,
        cardinality_threshold=100,
        partial_dependency_plot=10,
        shap_frac=0.05,
        importance_cutoff=0.00,
        corr_cutoff=0.9,
        search_space=LGB_SEARCH_SPACE,
        tuning_iters=25,
        lgb_params={},
    ):
        """
        Args:
            output_dir_path (string):  filepath where outputs package is created and saved
            csv_path (string): filepath to input csv
            target_col (string, optional): target column, default 'target'
            target_mapping (dict: optional): mapping of target col to binary value, default no-change 
            drop_cols (list: optional): columns to be dropped from target csv, default None 
            date_cols (list: optional): columns to be parsed as datetime then converted to numeric, default None 
            ord_cols_mapping= (dict: optional): columns to be converted to ordinal values, default None
            eda_flag (boolean, optional): EDA plots to be generates, default True
            tune_flag (boolean, optional): Lightgbm hyperparameters to be tuned, default True
            cardinality_threshold (numeric, optional): column cardinality determining one-hot-encoding or count-encoding, default 100
            partial_dependency_plot (numeric, optional): Generate SHAP dependency plots for N most important features, default 10
            shap_frac (numeric, optional): Proportion of data sampled for SHAP analysis, default 5%
            importance_cutoff (numeric, optional): Abs. avg. SHAP value below which feature is dropped, default 0.00
            corr_cutoff (numeric, optional): Abs. avg. correlation with more important feature above which feature is dropped, default 0.9
            search_space (numeric, optional): Tuning space for Bayesian optimisation, default is SKOPT_SEARCH_SPACE
            tuning_iter (numeric, optional): number of tuning iterations for Bayesian optimisation, default is 25,
            lgb_params (dict, optional): Hyperparams to use in case when tune_flag = False, default None
        """
        self.output_dir_path = output_dir_path
        self.csv_path = csv_path
        self.target_col = target_col
        self.target_mapping = target_mapping
        self.drop_cols = drop_cols
        self.date_cols = date_cols
        self.ord_cols_mapping = ord_cols_mapping
        self.eda_flag = eda_flag
        self.tune_flag = tune_flag
        self.cardinality_threshold = cardinality_threshold
        self.partial_dependency_plot = partial_dependency_plot
        self.shap_frac = shap_frac
        self.importance_cutoff = importance_cutoff
        self.corr_cutoff = corr_cutoff
        self.search_space = search_space
        self.tuning_iters = tuning_iters
        self.lgb_params = lgb_params

    def _gen_model_dir(self):
        """
        Creates output directory according to self.output_dir_path, removing previous output if there.

        Also makes subdirectories
            /bin
            /plots
        """
        logger.info(f"building directory {self.csv_path}")
        if os.path.exists(self.output_dir_path) and os.path.isdir(self.output_dir_path):
            shutil.rmtree(self.output_dir_path)
        os.mkdir(self.output_dir_path)
        os.mkdir(self.output_dir_path + "/bin")
        os.mkdir(self.output_dir_path + "/plots")

    def _process_csv(self):
        """
        Parses csv specifdiend in self.csv_path

        Also
            - parses 'date' columns as dates
            - drops 'ignore' columns
            - drops observations without valid 'target' values
            - Seperates features from target
            - Generates train and test set
        """
        logger.info(f"loading file {self.csv_path}")
        raw = pd.read_csv(
            self.csv_path, low_memory=False, parse_dates=self.date_cols
        ).drop(columns=self.drop_cols)

        raw.to_csv(f"{self.output_dir_path}/bin/raw.csv")

        X = raw.loc[raw[self.target_col].isin(self.target_mapping.keys()), :]
        y = X[self.target_col].map(self.target_mapping)
        X = X.drop(columns=self.target_col)
        del raw

        logger.info("train test split")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        del X, y

    def _encode_non_numeric_cols(self):
        """
        Encodes X_train according to following logic:
            - apply any specified ordinality maappings
            - map datetime features to numeric, also recording hour, and day of year
            - one-hot-encode text features with cardinality below threshold, filling in missing columns in 'transform' method
            - count-encode text features with cardinality above threshold, also recording lenght of string
        """
        logger.info("creating autoencoder")

        self.auto_encoder = preprocess.AutoEncoder(
            ordinality_mapping=self.ord_cols_mapping,
            date_cols=self.date_cols,
            cardinality_threshold=self.cardinality_threshold,
        )
        self.X_train_encoded = self.auto_encoder.fit_transform(self.X_train)
        self.X_test_encoded = self.auto_encoder.transform(self.X_test)

    def _tune(self):
        """
        Explores tuning space, updating self.lgb_params with values that minimize cross-validated brier score
        """
        # todo, can I save memory, code and possibly tune binning strats by passing unencoded X_train into pipeline?
        logger.info(f"tuning {self.tuning_iters}")
        results = bayes_hyperparam_tune(
            X=self.X_train_encoded,
            y=self.y_train,
            search_space=self.search_space,
            n_iters=self.tuning_iters,
        )
        self.lgb_params = results.best_params_
        logger.info(f"best params {self.lgb_params}")

    def _create_pipeline(self):
        """
        Creates sklearn pipeline with the following components
            - encoder: feature-encoder using strategy outlined in self._encode_non_numeric_cols
            - feature-selector: removes features with 'Shap' importance below threshold, or correlation above threshold with more important feature 
        """
        encoder = preprocess.AutoEncoder(
            ordinality_mapping=self.ord_cols_mapping,
            date_cols=self.date_cols,
            cardinality_threshold=self.cardinality_threshold,
        )
        feature_selector = preprocess.ColumnRemover(self.cols_to_remove)

        lgb_clf = lgb.LGBMClassifier(**self.lgb_params)

        self.classifer_pipeline = Pipeline(
            [
                ("encoder", encoder),
                ("feature selector", feature_selector),
                ("lgb classifier", lgb_clf),
            ]
        )
        self.classifer_pipeline.fit(self.X_train, self.y_train)

    def _save_model(self):
        """
        Saves sklearn pipeline as pkl and pmml files, also saves training file

        Args:
            pipeline (lightgbm pipeline) model to be saved
            output_dir (string): path to save model outputs
            train (df): dataset to save

        TODO - create E2E sklearn pipeline so that can be exported as PMML
        """
        output_dir = self.output_dir_path + "/bin"
        pmml_path = f"{output_dir}/model-pmml.pmml"
        pkl_path = f"{output_dir}/model-bin.pkl"
        training_data_path = f"{output_dir}/train.csv"

        data_train = self.X_train.copy()
        data_train["target"] = self.y_train

        data_train.to_csv(training_data_path, index=False)

        pickle.dump(self.classifer_pipeline, open(pkl_path, "wb"))

        # Annoying can't get PMML to work for entire pipeline - for now workaround fitting final model and exporting
        X_train_reduced = self.X_train_encoded.drop(columns=self.cols_to_remove)
        lgb_pmml = Pipeline([("lgb", lgb.LGBMClassifier(**self.lgb_params))])
        lgb_pmml.fit(X_train_reduced, self.y_train)
        features = X_train_reduced.columns
        target = "target"

        lgb_to_pmml(lgb_pmml, features, target, pmml_path)

    def auto_build(self):
        """
        Populates output_dir path, with model artifacts, and evalution charts
        """
        self._gen_model_dir()

        self._process_csv()

        if self.eda_flag:
            logger.info("EDA")
            general.dataset_eda(data=self.X_train, output_dir=self.output_dir_path)

        self._encode_non_numeric_cols()

        if self.tune_flag:
            self._tune()

        logger.info("fitting model")
        clf = lgb.LGBMClassifier(**self.lgb_params)
        clf.fit(self.X_train_encoded, self.y_train)

        logger.info("Assessing model")

        y_pred = clf.predict_proba(self.X_test_encoded)[:, 1]
        y_bm = np.repeat(self.y_train.mean(), self.y_test.shape)
        general.evaluate_model(
            self.y_test, y_pred, y_bm, self.output_dir_path, "Model - all features"
        )

        self.feature_importance = general.create_shap_plots(
            clf,
            self.X_train_encoded,
            output_dir=self.output_dir_path,
            N=self.partial_dependency_plot,
            frac=self.shap_frac,
        )

        logger.info("finding columns to remove")
        self.cols_to_remove = general.find_features_to_remove(
            importance=self.feature_importance,
            X=self.X_train_encoded,
            importance_cutoff=self.importance_cutoff,
            corr_threshold=self.corr_cutoff,
        )

        logger.info("fitting new model")
        self._create_pipeline()

        y_pred_reduced = self.classifer_pipeline.predict_proba(self.X_test)[:, 1]

        general.evaluate_model(
            self.y_test,
            y_pred_reduced,
            y_bm,
            self.output_dir_path,
            "Model - reduced features",
        )

        logger.info(f"saving model \n{self.output_dir_path}")

        self._save_model()
        logger.info("done!")


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
