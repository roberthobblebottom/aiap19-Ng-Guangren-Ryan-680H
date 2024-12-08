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


def column_rename_and_ensuring_consistency_values(df: pl.DataFrame):
    # rename column names
    column_names = [
        x.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("_Sensor", "")
        .lower()
        for x in df.columns
    ]
    column_names[0] = "location"
    column_names[4] = column_names[4][:-2] + "celsius"
    column_names[5] = column_names[5][:-1] + "percent"
    column_names[8] = column_names[8][:-2] + "m"
    df.columns = column_names

    # filter for consistency in the values, it is not need in the cross validation process so it is not in the sklearn pipeline.
    df = (
        df.with_columns(
            pl.col("nutrient_n_ppm").str.replace(" ppm", "").cast(pl.Float64),
            pl.col("nutrient_p_ppm").str.replace(" ppm", "").cast(pl.Float64),
            pl.col("nutrient_k_ppm").str.replace(" ppm", "").cast(pl.Float64),
            pl.col(
                "location",
                "previous_cycle_plant_type",
                "plant_type",
                "plant_stage",
            )
            .str.to_lowercase()
            .str.replace(" ", "_"),
        )
        .with_columns(
            plant_stage_coded=(
                pl.when(pl.col("plant_stage") == "seedling")
                .then(pl.lit(1))
                .when(pl.col("plant_stage") == "vegetative")
                .then(pl.lit(2))
                .when(pl.col("plant_stage") == "maturity")
                .then(pl.lit(3))
            ).cast(pl.Int8),
            plant_type_stage=pl.concat_str(
                pl.col("plant_type"), pl.lit("_"), pl.col("plant_stage")
            ),
        )
        .unique()  # remove duplicates
    )

    return df


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
            plant_type=pl.when(
                ~pl.col("plant_type").is_in(
                    ["fruiting_vegetables", "herbs", "leafy_greens", "vine_crops"]
                )
            )
            .then(pl.lit("None"))
            .otherwise(pl.col("plant_type")),
            plant_stage=pl.when(
                ~pl.col("plant_stage").is_in(["seedling", "vegetative", "maturity"])
            )
            .then(pl.lit("None"))
            .otherwise(pl.col("plant_stage")),
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
            humidity_percent=pl.when(~pl.col("humidity_percent").is_between(0, 100))
            .then(pl.lit(None))
            .otherwise(pl.col("humidity_percent")),
        )
        X = (
            X.with_columns(
                pl.when(pl.col("temperature_celsius") < 0)
                .then(pl.lit(None))
                .otherwise(pl.col("temperature_celsius"))
                .alias("temperature_celsius")
            )
            if self.include_temperature
            else X.with_columns(
                plant_type_stage=pl.when(
                    ~pl.col("plant_type_stage").is_in(
                        [
                            "fruiting_vegetables_seedling",
                            "fruiting_vegetables_vegetative",
                            "fruiting_vegetables_maturity",
                            "herbs_seedling",
                            "herbs_vegetative",
                            "herbs_maturity",
                            "leafy_greens_seedling",
                            "leafy_greens_vegetative",
                            "leafy_greens",
                            "vine_crops_seedling",
                            "vine_crops_vegetative",
                            "vine_crops_maturity",
                        ]
                    )
                )
                .then(pl.lit("None"))
                .otherwise(pl.col("plant_type_stage")),
            )
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
