# import marimo as mo
import polars as pl

import missingno

import numpy as np
from missforest import MissForest

from sklearn.model_selection import train_test_split, KFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


import statsmodels.formula.api as smf
import statsmodels.api as sm
import os

import Utils

if not os.path.exists("images"):
    os.mkdir("images")


class Temperature:
    def __init__(self, df: pl.DataFrame) -> None:
        self.kFold = KFold(n_splits=5)

        # if df.schema != pl.Schema(
        #     {
        #         "location": pl.String,
        #         "previous_cycle_plant_type": pl.String,
        #         "plant_type": pl.String,
        #         "plant_stage": pl.String,
        #         "temperature_celsius": pl.Float64,
        #         "humidity_percent": pl.Float64,
        #         "light_intensity_lux": pl.Float64,
        #         "co2_ppm": pl.Int64,
        #         "ec_dsm": pl.Float64,
        #         "o2_ppm": pl.Int64,
        #         "nutrient_n_ppm": pl.Float64,
        #         "nutrient_p_ppm": pl.Float64,
        #         "nutrient_k_ppm": pl.Float64,
        #         "ph": pl.Float64,
        #         "water_level_mm": pl.Float64,
        #     }
        # ):
        #     raise Exception(
        #         "Input Polars DataFrame is not in the correct schema configuration."
        #     )

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
                ).str.to_lowercase(),
                # pl.when(pl.col("previous_cycle_plant_type") == pl.col("plant_type"))
                # .then(pl.lit(0))
                # .otherwise(pl.lit(1))
                # .alias("plant_type_changed")
                # .cast(pl.Int8),
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
                    pl.col("plant_type"), pl.lit(" "), pl.col("plant_stage")
                ),
            )
            .unique()  # remove duplicates
        )
        if df.shape[0] == 0:
            raise Exception("There are no rows in the input Polars DataFrame.")
        df = df.filter(pl.col("temperature_celsius").is_not_nan())
        if df.shape[0] == 0:
            raise Exception(
                "There are no rows with temperature column that are not NaN values."
            )

        self.x = df.select(pl.exclude("temperature_celsius")).to_pandas()
        self.y = df.select("temperature_celsius").to_pandas()

        column_transformer = ColumnTransformer(
            [
                (
                    "oneHotEncoder",
                    OneHotEncoder(),
                    [
                        # "location",
                        # "previous_cycle_plant_type",
                        "plant_type",
                        "plant_stage",
                        "plant_type_stage",
                    ],
                ),
                (
                    "passthrough",
                    "passthrough",
                    [
                        # "temperature_celsius",
                        "humidity_percent",
                        "light_intensity_lux",
                        "co2_ppm",
                        "ec_dsm",
                        "o2_ppm",
                        "nutrient_n_ppm",
                        "nutrient_p_ppm",
                        "nutrient_k_ppm",
                        "ph",
                        "water_level_mm",
                        # "plant_type_changed",
                        # "plant_stage_coded",
                    ],
                ),
            ]
        )
        nonImportantFeatures = [
            [
                "plant_stage_coded",
                "previous_cycle_plant_type",
                "location",
                # "plant_type", # removed from here because it is still needed
            ]
        ]
        self.pipeline = Pipeline(
            [
                (
                    "nonImportantFeaturesRemover",
                    Utils.NonImportantFeaturesRemover(
                        nonImportantFeatures,
                        # list(self.x.columns)
                    ),
                ),
                ("outliersRemover", Utils.OutliersRemover()),
                ("columnTransformerForOneHotEncoding", column_transformer),
                (
                    "simpleImputer",
                    SimpleImputer(strategy="median"),
                ),  # MissForest was not working because it kept detecting temperature
                # even though I removed it from features.
                # replaced with Simpleimputer for speed and availability
                ("randomForestRegressor", RandomForestRegressor()),
            ]
        )

    def train(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.1
        )

        # # Hyperparam tuning with 5 fold LOOCV
        for x_indices, y_indices in self.kFold.split(x_train, x_test):

            self.pipeline.fit(x_train, y_train)
            self.pipeline.train()

    def evaluate(self):
        pass


if __name__ == "__main__":
    db = Utils.connect_sqlite("agri.db")
    Temperature(db).train()
