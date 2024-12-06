import sqlite3
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin


def connect_sqlite(db_path: str) -> pl.DataFrame:
    """Connects to sqlite3 database and return a polars DataFrame

    Args:
        db_path (str): path to sqlite database

    Raises:
        Exception: Exception risen when empty sqlite path given.

    Returns:
        pl.DataFrame: dataframe object
    """
    if db_path == "":
        raise Exception("Empty sqlite path given")
    connection = sqlite3.connect(db_path)
    return pl.read_database("SELECT * FROM farm_data", connection=connection)


class OutliersRemover(ClassifierMixin, BaseEstimator):
    """Removes outliers as explored in eda.ipynb"""

    def __init__(self, include_temperature=False):
        """

        Args:
            include_temperature (bool, optional): If outilier should includes temperature,
                                                  for plant_type_stage classifier Defaults to False.
        """
        self.include_temperature = include_temperature

    def fit(self, X, y):
        return self

    def transform(self, X):
        """The main function that removes the outliers by changing them into NaN values.

        Args:
            X (pl.DataFrame): features

        Returns:
            pl.DataFrame: features DataFrame with outliers removed
        """
        X = pl.from_pandas(X).with_columns(
            o2_ppm=pl.when(
                pl.col("plant_type") == pl.lit("fruiting_vegetables"),
                ~pl.col("o2_ppm").is_between(5, 8),
            )
            .then(pl.lit(None))
            .when(
                pl.col("plant_type") == pl.lit("herbs"),
                ~pl.col("o2_ppm").is_between(5, 8),
            )
            .then(pl.lit(None))
            .otherwise(pl.col("o2_ppm")),
            light_intensity_lux=pl.when(pl.col("light_intensity_lux") < 0)
            .then(pl.lit(None))
            .otherwise(pl.col("light_intensity_lux")),
            ec_dsm=pl.when(pl.col("ec_dsm") < 0)
            .then(pl.lit(None))
            .otherwise(pl.col("ec_dsm")),
        )
        X = (
            X.with_columns(
                pl.when(pl.col("temperature_celsius") < 0)
                .then(pl.lit(None))
                .otherwise(pl.col("temperature_celsius"))
                .alias("temperature_celsius")
            )
            if self.include_temperature
            else X
        ).to_pandas()

        return X


class NonImportantFeaturesRemover(ClassifierMixin, BaseEstimator):
    """Removes non important feaetures as mentioned in th eda"""

    def __init__(
        self,
        features_to_remove: list[str],
    ):
        self.features_to_remove = features_to_remove

    def fit(self, x, y):
        return self

    def transform(self, x):
        """main function in  removing non important features"""
        x = pl.from_pandas(x)
        x = x.select(pl.exclude(*self.features_to_remove)).to_pandas()
        return x
