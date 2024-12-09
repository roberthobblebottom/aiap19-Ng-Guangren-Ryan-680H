import polars as pl

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import Utils
from scipy.stats import randint


class Plant_Type_Stage:
    def __init__(self, df: pl.DataFrame) -> None:
        print("Plant_type_stage.py init")
        df = Utils.column_rename_and_ensuring_consistency_values(df)
        nonImportantFeatures = [
            "plant_stage_coded",
            "previous_cycle_plant_type",
            "location",
            "nutrient_p_ppm",
            "nutrient_n_ppm",  # these two nutrients are removed as discussed in the eda.
            "plant_type",
        ]
        # Some guards to prevent incorrect inputs or formats from being trained and tested
        if df.shape[0] == 0:
            raise Exception("There are no rows in the input Polars DataFrame.")

        # Excluding non important features as mentioned in the eda out of the sklearn pipeline seems to work better
        self.x = (
            df.select(pl.exclude("plant_type_stage", "plant_stage"))
            .select(pl.exclude(nonImportantFeatures))
            .to_pandas()
        )
        self.y = df.select("plant_type_stage").to_pandas()

        # column_transformer = ColumnTransformer(
        #     [
        #         (
        #             "oneHotEncoder",
        #             OneHotEncoder(),
        #             [
        #                 "plant_type",
        #                 "plant_stage",
        #             ],
        #         ),
        #         (
        #             "passthrough",
        #             "passthrough",
        #             [
        #                 "temperature_celsius",
        #                 "humidity_percent",
        #                 "light_intensity_lux",
        #                 "co2_ppm",
        #                 "ec_dsm",
        #                 "o2_ppm",
        #                 "nutrient_k_ppm",
        #                 "ph",
        #                 "water_level_mm",
        #             ],
        #         ),
        #     ]
        # )

        """
         ['previous_cycle_plant_type_vine crops', 'previous_cycle_plant_type_fruiting vegetables', 'previous_cycle_plant_type_herbs', 
         'previous_cycle_plant_type_leafy greens', 'location_zone_e', 'location_zone_c',
           'location_zone_f', 'location_zone_g', 'location_zone_b', 'location_zone_d', 'location_zone_a', 'plant_type_changed']
        """
        self.pipeline = Pipeline(
            [
                # (
                #     "nonImportantFeaturesRemover",
                #     Utils.NonImportantFeaturesRemover(
                #         nonImportantFeatures,
                #     ),
                # ),
                ("outliersRemover", Utils.OutliersRemover(is_classification_task=True)),
                # (
                #     "columnTransformerForOneHotEncoding",
                #     column_transformer,
                # ),
                # (
                #     "simpleImputer",
                #     SimpleImputer(strategy="median"),
                # ),  # MissForest was not working because it kept detecting temperature
                # even though I removed it from features.
                # replaced with Simpleimputer for speed and availability
                (
                    "randomForestClassifier",
                    RandomForestClassifier(
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
            "randomForestClassifier__n_estimators": randint(10, 200),
            "randomForestClassifier__max_depth": randint(1, 20),
            "randomForestClassifier__min_samples_split": randint(2, 11),
            "randomForestClassifier__min_samples_leaf": randint(1, 11),
        }
        rs = RandomizedSearchCV(
            self.pipeline,
            param_distributions=param_dist,
            n_iter=20,  # TODO change to a bigger number
            cv=5,
            scoring="accuracy",
            n_jobs=20,
        )
        rs.fit(x_train, y_train.iloc[:, 0].ravel())
        best_params = rs.best_params_
        print("best_params", best_params)

        print("training model")
        self.pipeline.set_params(
            **{
                "randomForestClassifier__n_estimators": best_params[
                    "randomForestClassifier__n_estimators"
                ],
                "randomForestClassifier__max_depth": best_params[
                    "randomForestClassifier__max_depth"
                ],
                "randomForestClassifier__min_samples_split": best_params[
                    "randomForestClassifier__min_samples_split"
                ],
                "randomForestClassifier__min_samples_leaf": best_params[
                    "randomForestClassifier__min_samples_leaf"
                ],
            }
        )
        self.pipeline.fit(x_train, y_train.iloc[:, 0].ravel())

    def evaluate(self):
        """
        Rationale for metrics used.
        Accuracy: Useful for looking at all classes.
        F1: Balances between precision and recall; the harmonic mean between the two.
            Useful if the aim is to look at the positives classes only
        AUC-ROC: prvides insights into the performance of the model
        """
        print("evaluations")
        predictions = self.pipeline.predict(self.x_test)
        predictions_probabilites = self.pipeline.predict_proba(self.x_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average="macro")
        roc_auc = roc_auc_score(
            label_binarize(self.y_test, classes=self.pipeline.classes_),
            predictions_probabilites,
            multi_class="ovr",
        )
        print("accuracy", accuracy)
        print("f1", f1)
        print("roc auc", roc_auc)


if __name__ == "__main__":
    db = Utils.connect_sqlite("agri.db")
    t = Plant_Type_Stage(db)
    t.train()
    t.evaluate()
