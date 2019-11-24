import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import subprocess
import pickle
from sklearn.pipeline import Pipeline
import shutil
import os
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from nyoka import skl_to_pmml, lgb_to_pmml

import config
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
pd.options.mode.chained_assignment = None  # default='warn'


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
    
    Builds binary classifier, including:
        - dataset EDA (optional)
        - hyperparameter tuning (optional)
        - model performance assessment
        - SHAP-based feature analysis
        - feature selection
        - creating deployment package (pmml & pkl)

    Attributes:
        auto_build (method): automatically builds bin populates output_dir path, with model artifacts, and evaluation charts

    """

    def __init__(
        self,
        output_dir_path,
        csv_path,
        target_col="target",
        ignore_cols=[],
        eda_flag=True,
        tune_flag=True,
        cardinality_threshold=100,
        shap_plot_num=10,
        shap_frac=0.05,
        importance_cutoff=0.00,
        corr_cutoff=0.9,
        search_space=LGB_SEARCH_SPACE,
        tuning_iters=25,
        lgb_params={},
        random_state=1234,
    ):
        """
        Args:
            output_dir_path (string):  filepath where outputs package is created and saved
            csv_path (string): filepath to input csv, NOTE need to preprocess columns to be numeric or string type
            target_col (string, optional): target column, default 'target'
            ignore_cols (iterable, optional): columns to be dropped, default []
            eda_flag (boolean, optional): EDA plots to be generated, default True
            tune_flag (boolean, optional): Lightgbm hyperparameters to be tuned, default True
            shap_plot_num (numeric, optional): Generate SHAP dependency plots for N most important features, default 10
            shap_frac (numeric, optional): Proportion of data sampled for SHAP analysis, default 5%
            importance_cutoff (numeric, optional): Abs. avg. SHAP value threshold suggest dropping feature, default 0.00
            corr_cutoff (numeric, optional): Abs. avg. correlation suggest dropping feature, default 0.9
            search_space (numeric, optional): Tuning space for Bayesian optimisation, default is SKOPT_SEARCH_SPACE
            tuning_iter (numeric, optional): number of tuning iterations for Bayesian optimisation, default is 25,
            lgb_params (dict, optional): Hyperparams to use in case when tune_flag = False, default None
            random_state (numeric, optional): Random seed for train test split, and model-training - default is 1234
        """
        self.output_dir_path = output_dir_path
        self.csv_path = csv_path
        self.target_col = target_col
        self.ignore_cols = ignore_cols
        self.eda_flag = eda_flag
        self.tune_flag = tune_flag
        self.cardinality_threshold = cardinality_threshold
        self.shap_plot_num = shap_plot_num
        self.shap_frac = shap_frac
        self.importance_cutoff = importance_cutoff
        self.corr_cutoff = corr_cutoff
        self.search_space = search_space
        self.tuning_iters = tuning_iters
        self.lgb_params = lgb_params
        self.random_state = random_state

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
        Parses csv specified in self.csv_path, saving to self.raw

        Also
            - Drops ignore columns
            - Validates target and feature columns
                Target = binary, 0-1
                Features = numeric or string
        """
        logger.info(f"loading file {self.csv_path}")
        raw = pd.read_csv(self.csv_path).drop(columns=self.ignore_cols)

        logger.info("checking valid input data")
        assert raw[self.target_col].isna().sum() == 0

        assert list(sorted(raw[self.target_col].unique())) == [0, 1]

        valid_shape = raw.select_dtypes(include=["int64", "float64", "object"]).shape
        assert valid_shape == raw.shape
        self.raw = raw
        raw.to_csv(f"{self.output_dir_path}/bin/raw.csv")

    def _prepare_X_y(self):
        """
        Splits self raw into X_train y_train, X_test, y_test 

        Also records categorical and numerical columns, and saves csv of training set
        """

        y = self.raw[self.target_col]
        X = self.raw.drop(columns=self.target_col)

        logger.info("train test split")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state
        )
        data_train = self.X_train.copy()
        data_train["target"] = self.y_train

        training_data_path = f"{self.output_dir_path}/bin/train.csv"
        data_train.to_csv(training_data_path, index=False)

        del X, y

    def _create_categorical_transformer(self):
        self.categorical_cols = self.X_train.select_dtypes(include=["object"]).columns
        self.numeric_cols = self.X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns

        self.mapper = DataFrameMapper(
            [
                ([cat_column], [CategoricalDomain(), LabelEncoder()])
                for cat_column in self.categorical_cols
            ]
            + [(self.numeric_cols, ContinuousDomain())]
        )

        # hacky, also storing seperated X_train_encoded and classifier, because couldn't get SHAP and skopt to work for e2e pipeline
        self.X_train_encoded = self.mapper.fit_transform(self.X_train)
        self.var_names = self.X_train.columns

    def _tune(self):
        """
        Explores tuning space, updating self.lgb_params with values that minimize cross-validated brier score
        """
        # todo, can I save memory, code and possibly tune binning strats by passing unencoded X_train into pipeline?
        logger.info(f"tuning {self.tuning_iters}")
        results = utils.bayes_hyperparam_tune(
            model=lgb.LGBMClassifier(objective="binary"),
            X=self.X_train_encoded,
            y=self.y_train,
            search_space=self.search_space,
            n_iters=self.tuning_iters,
        )
        self.lgb_params = results.best_params_
        logger.info(f"best params {self.lgb_params}")

    def _save_model(self):
        """
        Saves sklearn pipeline as pkl and pmml files, also saves training file

        Args:
            pipeline (lightgbm pipeline) model to be saved
            output_dir (string): path to save model outputs
            train (df): dataset to save
        """
        pmml_path = f"{self.output_dir_path}/model-pmml.pmml"
        pkl_path = f"{self.output_dir_path}/model-bin.pkl"
        pickle.dump(self.pipeline, open(pkl_path, "wb"))
        # sklearn2pmml(self.pipeline, pmml_path)

    def _generate_shap_plots(self):
        classifier = lgb.LGBMClassifier(**self.lgb_params)
        classifier.fit(self.X_train_encoded, self.y_train)
        X_shap = pd.DataFrame(data=self.X_train_encoded, columns=self.var_names)
        self.feature_importance = utils.create_shap_plots(
            classifier,
            X_shap,
            output_dir=self.output_dir_path,
            N=self.shap_plot_num,
            frac=self.shap_frac,
        )

    def auto_build(self):
        """
        Populates output_dir path, with model artifacts, and evalution charts
        """
        self._gen_model_dir()

        self._process_csv()

        self._prepare_X_y()

        if self.eda_flag:
            logger.info("EDA")
            utils.dataset_eda(data=self.X_train, output_dir=self.output_dir_path)

        self._create_categorical_transformer()

        if self.tune_flag:
            self._tune()

        self._generate_shap_plots()

        logger.info("creating pipeline")
        classifier = lgb.LGBMClassifier(**self.lgb_params)
        self.pipeline = PMMLPipeline(
            [("mapper", self.mapper), ("classifier", classifier)]
        )

        self.pipeline.fit(self.X_train, self.y_train)

        logger.info("Assessing model")

        y_pred = self.pipeline.predict_proba(self.X_test)[:, 1]
        y_bm = np.repeat(self.y_train.mean(), self.y_test.shape)
        utils.evaluate_model(self.y_test, y_pred, y_bm, self.output_dir_path, "Model")

        logger.info("suggeting features to remove")
        self.cols_to_remove = utils.find_features_to_remove(
            importance=self.feature_importance,
            X=self.X_train,
            importance_cutoff=self.importance_cutoff,
            corr_threshold=self.corr_cutoff,
        )
        logger.info(f"candidates to remove - {self.cols_to_remove}")

        logger.info(f"saving model \n{self.output_dir_path}")

        self._save_model()
        test_input = dict(self.X_test.iloc[0])
        test_score = self.pipeline.predict_proba(self.X_test.head(1))
        logger.info(
            f"test-case model inputs \n{ test_input } \n model score \n {test_score}"
        )

        logger.info("done!")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    auto_builder = AutoBuilder(
        csv_path=config.csv_path,
        output_dir_path=config.output_dir_path,
        target_col=config.target_col,
        ignore_cols=config.ignore_cols,
        tuning_iters=25,
        tune_flag=True,
        eda_flag=True,
    )
    auto_builder.auto_build()
