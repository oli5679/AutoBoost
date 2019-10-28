import sys
import config
import logging

from utils import auto_builder


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    auto_builder = auto_builder.AutoBuilder(
        csv_path=config.csv_path,
        output_dir_path=config.output_dir_path,
        target_col=config.target_col,
        drop_cols=config.drop_cols,
        date_cols=config.date_cols,
        tuning_iters=25,
    )
    auto_builder.auto_build()
