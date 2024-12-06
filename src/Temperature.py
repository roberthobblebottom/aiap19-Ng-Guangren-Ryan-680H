import polars as pl

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

import Utils
from scipy.stats import randint


class Temperature:
    def __init__(self, df: pl.DataFrame) -> None:

        # I failed in this one
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

        # Some guards to prevent incorrect inputs or formats from being trained and tested
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
                (
                    "randomForestRegressor",
                    RandomForestRegressor(
                        n_jobs=10,
                        random_state=0,
                    ),
                ),
            ]
        )

    def train(self):
        """
        3 way hold out with cross validation it is used because
        unlike 2 way hold out, the test error less depended on samples use
        and may overestimate test error

        these split is used in such a way that data leakage isn't present.
        Data Leakage will give a overestimated test error.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.1
        )

        # These test splits are for evaluation later
        self.x_test = x_test
        self.y_test = y_test

        """
         For hyperparmater tuning. Faster than grid search or bayesian optimisation. 
         includes 5 fold cross validation:
        """
        print("hyperparameter tuning")
        param_dist = {
            "randomForestRegressor__n_estimators": randint(10, 200),
            "randomForestRegressor__max_depth": randint(1, 20),
            "randomForestRegressor__min_samples_split": randint(2, 11),
            "randomForestRegressor__min_samples_leaf": randint(1, 11),
        }
        rs = RandomizedSearchCV(
            self.pipeline,
            param_distributions=param_dist,
            n_iter=10,  # TODO change to a bigger number
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=10,
        )
        rs.fit(x_train, y_train.iloc[:, 0].ravel())
        best_params = rs.best_params_
        print("best_params", best_params)

        print("training model")
        self.pipeline.set_params(
            {
                "randomForestRegressor__n_estimators": best_params[
                    "randomForestRegressor__n_estimators"
                ],
                "randomForestRegressor__max_depth": best_params[
                    "randomForestRegressor__max_depth"
                ],
                "randomForestRegressor__min_samples_split": best_params[
                    "randomForestRegressor__min_samples_split"
                ],
                "randomForestRegressor__min_samples_leaf": best_params[
                    "randomForestRegressor__min_sample_leaf"
                ],
            }
        )
        self.pipeline.fit(x_train, y_train)

    def evaluate(self):
        """
        Rationale for metrics used.
        Mean Absolute Error: Rombust to outliers,Interpretability
        Root Mean Squared Error: More interpretable than mean squared error, Less sensitive to large errors
        r2: Proportion of variance explained, Scale indenpendce compare to the other regression metrics, ease of intepretability
        """
        # predictions = self.pipeline.predict(self.x_test)
        prediction_probabilities = self.pipeline.predict_proba(self.x_test)
        r2 = r2_score(self.y_test, prediction_probabilities)
        mae = mean_absolute_error(self.y_test, prediction_probabilities)
        rmse = root_mean_squared_error(self.y_test, prediction_probabilities)
        print("r square score", r2)
        print("mean absolute error", mae)
        print("root mean squared error", rmse)


if __name__ == "__main__":
    db = Utils.connect_sqlite("agri.db")
    t = Temperature(db)
    t.train()
    t.evaluate()
