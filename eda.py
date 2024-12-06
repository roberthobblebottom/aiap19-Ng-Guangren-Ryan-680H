import marimo

__generated_with = "0.9.30"
app = marimo.App(width="full", app_title="aiap19", auto_download=["html"])


@app.cell
def __(mo):
    mo.md(r"""#Imports""")
    return


@app.cell
def __():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import missingno
    import sqlite3
    import numpy as np
    from missforest import MissForest

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
    from sklearn.feature_selection import SelectKBest

    from scipy.stats import shapiro, levene, bartlett, ttest_1samp
    from scipy.stats.contingency import association


    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import os

    if not os.path.exists("images"):
        os.mkdir("images")
    return (
        DecisionTreeRegressor,
        MissForest,
        OneHotEncoder,
        PolynomialFeatures,
        RandomForestClassifier,
        RandomForestRegressor,
        SelectKBest,
        association,
        bartlett,
        levene,
        missingno,
        mo,
        np,
        os,
        pl,
        px,
        shapiro,
        sm,
        smf,
        sqlite3,
        train_test_split,
        ttest_1samp,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        # db Connection and renaming of columns
        ## Renaming column names into something more managable programmically.
        the word "sensor" is redundant.
        """
    )
    return


@app.cell
def __(pl, sqlite3):
    connection = sqlite3.connect("agri.db")
    df = pl.read_database("SELECT * FROM farm_data", connection=connection)

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
    df.columns
    return column_names, connection, df


@app.cell
def __(df):
    df.describe()
    return


@app.cell
def __(df):
    df["nutrient_n_ppm"].unique()
    return


@app.cell
def __(df):
    df["nutrient_p_ppm"].unique()
    return


@app.cell
def __(df):
    df["nutrient_k_ppm"].unique()
    return


@app.cell
def __(df):
    df["location"].unique()
    return


@app.cell
def __(df):
    df["previous_cycle_plant_type"].unique()
    return


@app.cell
def __(df):
    df["plant_type"].unique()
    return


@app.cell
def __():
    return


@app.cell
def __(df):
    df["plant_stage"].unique()
    return


@app.cell
def __(mo):
    mo.md(r"""### plant stages of seedling, vegetative and maturity, it is ordinal in nature""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ##Explaination of the code in the cell bellow:

        values of these features (nutrient_n_ppm, nutrient_p_ppm, nutrient_k_ppm) are not consistent. They will be casted be all of float type and the string " ppm" will be removed.


        Values of these features (location, previous_cycle_plant_type, plant_type, plant_stage) are of inconsistent capitalisation. they will be all lower caps

        plant stage ordinal encoding:
        seedling -> 1  
        vegetative -> 2  
        maturity -> 3

        if there is change from previous cycle plant types to current cycle pant types, plant_type_change encoding:  
        changed -> 1  
        not changed -> 0  

        unique() is for dropping duplicates
        """
    )
    return


@app.cell
def __(df, pl):
    df_consistent = (
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
            pl.when(pl.col("previous_cycle_plant_type") == pl.col("plant_type"))
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("plant_type_changed")
            .cast(pl.Int8),
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
        .unique()
    )
    return (df_consistent,)


@app.cell
def __(df_consistent):
    df_consistent.describe()
    return


@app.cell
def __(df):
    df.head(10)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### light_intensity_lux, ec_dsm, temperature_celsius has negatives. light intensity lux, ec_dsm simply can't go negative and crops won't grow in negative temperature. All these features's negatives will be imputed

        source for light_intensity_lux: table under this seciton: https://en.wikipedia.org/wiki/Lux#Illuminance
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        <!-- # Histograms

        note to self: most other histograms have same propotionality -->
        """
    )
    return


@app.cell
def __():
    # _f = px.histogram(
    #     df_consistent.to_pandas(),
    #     x="plant_type_changed",
    #     title="More plant types changes than plant types that stayed the same",
    # )
    # _f.update_layout(
    #     dragmode=False,  # Disable dragging
    #     hovermode=False,  # Disable hover info
    # )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Correlations heatmap and Scatterplot matrix

        many of the scatterplots has image caching because this notebook is getting too slow
        """
    )
    return


@app.cell
def __(df_consistent, px):
    px.imshow(
        df_consistent.to_pandas().corr().round(2),
        text_auto=True,
        title="pearson correlation heatmap",
    )
    return


@app.cell
def __(df_consistent, px):
    px.imshow(
        df_consistent.to_pandas().corr(method="spearman").round(2),
        text_auto=True,
        title="spearman correlation heatmap",
    )
    return


@app.cell
def __(mo):
    mo.md(r"""### used spearman just in case there are non linearities but seesm that both spearman and pearson shows similar correlations. then I only made scatterplots with higher correlations that is more than 0.6 or less than -0.6 because a bigger scatter plot lags the notebook. plant_type_changed is not as useful as it seems""")
    return


@app.cell
def __(df_consistent, mo, os, px):
    _names = [
        "humidity_percent",
        "light_intensity_lux",
        "nutrient_n_ppm",
        "nutrient_p_ppm",
        "nutrient_k_ppm",
        "plant_stage_coded",
    ]
    if not os.path.exists("images/scatterplot_matrix.png"):
        print("here")
        px.scatter_matrix(
            df_consistent.select(_names).to_pandas(),
        ).update_layout(width=1000, height=1000).write_image(
            "images/scatterplot_matrix.png"
        )
    mo.image("images/scatterplot_matrix.png")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ###Used this scatter plot matrix to check the overall patterns of features of interests to see what pair of features shold I investigated more closely via enlarged scatter plot with trendline and box plots

        #### nutrient p and n seems to have correlations to nutrient n and light intensity lux, p and k nutrients may be redundant as shown in the scatter plots matrix above and the following scatter plots below.

        ### the light_intensity_lux is very v shaped for the nutrients introduced and humidity as shown above and below.

        ### there are interactivities:  
        between the nutrients   
        between humidity and each of the nutrients but on a milder negative sense
        """
    )
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/scatterplot_n_lux.png"):
        px.scatter(
            df_consistent.to_pandas(),
            x="light_intensity_lux",
            y="nutrient_n_ppm",
            trendline="lowess",
            trendline_color_override="red",
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/scatterplot_n_lux.png")
    mo.image("images/scatterplot_n_lux.png")
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/scatterplot_p_lux.png"):
        px.scatter(
            df_consistent.to_pandas(),
            x="light_intensity_lux",
            y="nutrient_p_ppm",
            trendline="lowess",
            trendline_color_override="red",
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/scatterplot_p_lux.png")
    mo.image("images/scatterplot_p_lux.png")
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/scatterplot_humidity_p.png"):
        px.scatter(
            df_consistent.to_pandas(),
            x="humidity_percent",
            y="nutrient_k_ppm",
            trendline="lowess",
            trendline_color_override="red",
            title="slight negative trend of nutrient k and humidity",
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/scatterplot_humidity_p.png")
    mo.image("images/scatterplot_humidity_p.png")
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/scatterplot_k_p.png"):
        px.scatter(
            df_consistent.to_pandas(),
            x="nutrient_p_ppm",
            y="nutrient_k_ppm",
            trendline="lowess",
            trendline_color_override="red",
            title=" positive trend of nutrient k and nutrient p",
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/scatterplot_k_p.png")
    mo.image("images/scatterplot_k_p.png")
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/scatterplot_n_p.png"):
        px.scatter(
            df_consistent.to_pandas(),
            x="nutrient_p_ppm",
            y="nutrient_n_ppm",
            trendline="lowess",
            trendline_color_override="red",
            title="slight positive trend of nutrient n and nutrient p",
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/scatterplot_n_p.png")
    mo.image("images/scatterplot_n_p.png")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # box plots

        box plots are cached too
        """
    )
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/box_k_plant_stage.png"):
        px.box(
            df_consistent.to_pandas(), y="nutrient_k_ppm", x="plant_stage"
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/box_k_plant_stage.png")
    mo.image("images/box_k_plant_stage.png")
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/box_n_plant_stage.png"):
        px.box(
            df_consistent.to_pandas(), y="nutrient_n_ppm", x="plant_stage"
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/box_n_plant_stage.png")
    mo.image("images/box_n_plant_stage.png")
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/box_p_plant_stage.png"):
        px.box(
            df_consistent.to_pandas(), y="nutrient_p_ppm", x="plant_stage"
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/box_p_plant_stage.png")
    mo.image("images/box_p_plant_stage.png")
    return


@app.cell
def __(mo):
    mo.md(r"""### lower levels of nutrients at seedling stage and very similar proportionality for all three nutrients. this supports the notion that k and p nutrients may be redundant.""")
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/box_plant_type_temp.png"):
        px.box(
            df_consistent.to_pandas(),
            x="plant_type",
            y="temperature_celsius",
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/box_plant_type_temp.png")
    mo.image("images/box_plant_type_temp.png")
    return


@app.cell
def __():
    # os.system("rm -f images/box_plant_type_temp.png")
    return


@app.cell
def __(mo):
    mo.md(r"""To check I assume crops won't be able to survive such low negative temperatures, these outliers will be handled""")
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/box_plant_type_humidity.png"):
        px.box(
            df_consistent.to_pandas(),
            x="plant_type",
            y="humidity_percent",
            title="vine crops has higher 25th, 50th and 75t percentile points than the other plant types",
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/box_plant_type_humidity.png")
    mo.image("images/box_plant_type_humidity.png")
    return


@app.cell
def __(df_consistent, mo, os, px):
    if not os.path.exists("images/box_plant_type_lux.png"):
        px.box(
            df_consistent.to_pandas(),
            x="plant_type",
            y="light_intensity_lux",
            title="",
        ).update_layout(
            dragmode=False,  # Disable dragging
            hovermode=False,  # Disable hover info
        ).write_image("images/box_plant_type_lux.png")
    mo.image("images/box_plant_type_lux.png")
    return


@app.cell
def __(mo):
    mo.md("""there is no such thing as negative lux or luminus flux so these outliers has to be imputed""")
    return


@app.cell
def __(df_consistent, px):
    px.box(
        df_consistent.to_pandas(),
        x="plant_type",
        y="co2_ppm",
        title="herbs and fruiting vegetables co2 level is generally more than the other plant types",
    ).update_layout(
        dragmode=False,  # Disable dragging
        hovermode=False,  # Disable hover info
    )
    return


@app.cell
def __(df_consistent, px):
    px.violin(
        df_consistent.to_pandas(),
        x="plant_type",
        y="ec_dsm",
        title="other than the negatives, the outliers may paint a long tail picture, only negatives will be removed",
        box=True,
        points="outliers",
    ).update_layout(
        dragmode=False,  # Disable dragging
        hovermode=False,  # Disable hover info
    )
    return


@app.cell
def __(df_consistent, px):
    px.box(
        df_consistent.to_pandas(),
        x="plant_type",
        y="o2_ppm",
        title="there are little amount of outliers that is spread for herbs and fruiting vegetables to be a tail of a distribution so they will be removed",
    )
    return


@app.cell
def __(df_consistent, px):
    px.box(
        df_consistent.to_pandas(), x="plant_type", y="nutrient_n_ppm"
    ).update_layout(
        dragmode=False,  # Disable dragging
        hovermode=False,  # Disable hover info
    )
    return


@app.cell
def __(df_consistent, px):
    px.box(
        df_consistent.to_pandas(), x="plant_type", y="nutrient_p_ppm"
    ).update_layout(
        dragmode=False,  # Disable dragging
        hovermode=False,  # Disable hover info
    )
    return


@app.cell
def __(mo):
    mo.md(r"""### leafy greens has lower nutrient phosphorus and nitrogen values""")
    return


@app.cell
def __(df_consistent, px):
    px.box(
        df_consistent.to_pandas(), x="plant_type", y="nutrient_k_ppm"
    ).update_layout(
        dragmode=False,  # Disable dragging
        hovermode=False,  # Disable hover info
    )
    return


@app.cell
def __(mo):
    mo.md(r"""### vine crops has overall lowest nutrient k value, leafy green on the 1st quartitle comes next but leafy is also have a wide range.""")
    return


@app.cell
def __(df_consistent, px):
    px.violin(
        df_consistent.to_pandas(),
        x="plant_type",
        y="ph",
        box=True,
        points="outliers",
        title="these outliers represents longer tail, they will be kept",
    ).update_layout(
        dragmode=False,  # Disable dragging
        hovermode=False,  # Disable hover info
    )
    return


@app.cell
def __(mo):
    mo.md(r"""<!-- # certain feature's outliers will be set to NAN, and they will be nan imputed later -->""")
    return


@app.cell
def __():
    # df_less_outs = df_consistent.with_columns(
    #     pl.when(pl.col("temperature_celsius") < 0)
    #     .then(pl.lit(None))
    #     .otherwise(pl.col("temperature_celsius"))
    #     .alias("temperature_celsius"),
    #     o2_ppm=pl.when(
    #         pl.col("plant_type") == pl.lit("fruiting_vegetables"),
    #         ~pl.col("o2_ppm").is_between(
    #             5,
    #             8,
    #         ),
    #     )
    #     .then(pl.lit(None))
    #     .when(
    #         pl.col("plant_type") == pl.lit("herbs"),
    #         ~pl.col("o2_ppm").is_between(5, 8),
    #     )
    #     .then(pl.lit(None))
    #     .otherwise(pl.col("o2_ppm")),
    #     light_intensity_lux=pl.when(pl.col("light_intensity_lux") < 0)
    #     .then(pl.lit(None))
    #     .otherwise(pl.col("light_intensity_lux")),
    #     ec_dsm=pl.when(pl.col("ec_dsm") < 0)
    #     .then(pl.lit(None))
    #     .otherwise(pl.col("ec_dsm")),
    # )
    return


@app.cell
def __(mo):
    mo.md(r"""<!-- # Checking on NANs -->""")
    return


@app.cell
def __(df_consistent, missingno):
    missingno.matrix(df_consistent.to_pandas())
    return


@app.cell
def __(df_consistent, missingno):
    missingno.heatmap(df_consistent.to_pandas())
    return


@app.cell
def __(mo):
    mo.md(r"""## Data are missing at random""")
    return


@app.cell
def __(mo):
    mo.md(
        """
        # Getting feature importance for regression task
        sampling it because at this point of the notebook, it is taking too long to run
        """
    )
    return


@app.cell
def __(OneHotEncoder):
    e = OneHotEncoder(sparse_output=False).set_output(transform="polars")
    return (e,)


@app.cell
def __(RandomForestRegressor, df_consistent, e, pl):
    rfr = RandomForestRegressor(n_jobs=20)
    df_regression = df_consistent.filter(~pl.col("temperature_celsius").is_nan())
    df_one_hot_encoded_regression = e.fit_transform(
        df_regression.select(pl.col(pl.String()))
        .select(pl.exclude("plant_type_stage", "plant_stage"))
        .to_pandas()
    )
    _y = df_regression.select("temperature_celsius").to_series().to_pandas()
    df_regression = df_one_hot_encoded_regression.hstack(
        df_regression.select(pl.exclude(pl.String()))
    ).select(pl.exclude("temperature_celsius"))
    rfr.fit(df_regression, _y)
    feature_importance_regression = pl.DataFrame(
        [rfr.feature_names_in_, rfr.feature_importances_],
        schema=["name", "importance"],
    )
    return (
        df_one_hot_encoded_regression,
        df_regression,
        feature_importance_regression,
        rfr,
    )


@app.cell
def __(feature_importance_regression):
    print(
        "More important features for classification: \n",
        feature_importance_regression.sort("importance", descending=True)[
            :15, "name"
        ].to_list(),
    )
    return


@app.cell
def __(mo):
    mo.md("""## these features below will be manually removed in the ml pipeline""")
    return


@app.cell
def __(feature_importance_regression):
    print(
        "Less important features for classification: \n",
        feature_importance_regression.sort("importance", descending=True)[
            15:, "name"
        ].to_list(),
    )
    return


@app.cell
def __(mo):
    mo.md(r"""# Getting feature importance for classification task""")
    return


@app.cell
def __(df_consistent, pl):
    print(df_consistent.select(pl.exclude(pl.String())).columns)
    return


@app.cell
def __(RandomForestClassifier, df_consistent, e, pl):
    rfc = RandomForestClassifier(n_jobs=20)
    df_classification = df_consistent
    df_one_hot_encoded_classification = e.fit_transform(
        df_classification.select(pl.col(pl.String()))
        .select(pl.exclude("plant_type_stage", "plant_stage", "plant_stage_coded"))
        .to_pandas()
    )
    _y = df_classification.select("plant_type_stage").to_series().to_pandas()
    df_classification = df_one_hot_encoded_classification.hstack(
        df_classification.select(pl.exclude(pl.String()))
    ).select(pl.exclude("plant_stage_coded"))
    rfc.fit(df_classification, _y)
    feature_importance_classification = pl.DataFrame(
        [rfc.feature_names_in_, rfc.feature_importances_],
        schema=["name", "importance"],
    )
    return (
        df_classification,
        df_one_hot_encoded_classification,
        feature_importance_classification,
        rfc,
    )


@app.cell
def __(feature_importance_classification):
    print(
        "More important features for classification: \n",
        feature_importance_classification.sort("importance", descending=True)[
            :15, "name"
        ].to_list(),
    )
    return


@app.cell
def __(mo):
    mo.md("""## these features below will be manually removed in the ml pipeline""")
    return


@app.cell
def __(feature_importance_classification):
    print(
        "Less important features for classification: \n",
        feature_importance_classification.sort("importance", descending=True)[
            15:, "name"
        ].to_list(),
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # feature importances finding
        in both classficaiton and regression task, nutrient k has alot of important following nutrient n and p which backs up the box plots, sctter plots and correlation heatmaps that was previously discussed too  

        humidity is not as important just like the previous correlations heatmaps, scatter plots have suggested.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Hypothesis testing for classifcaiton against target

        ## Test of normality
        """
    )
    return


@app.cell
def __(MissForest, df_consistent, os, pl):
    missForestImputer = MissForest(
        max_iter=5
    )  # max iter to 3 because 5 iters took too long. but it may be less accurate

    if not os.path.exists("imputed_cache.csv") or True:
        print("a")
        df_imputed = missForestImputer.fit_transform(
            df_consistent.select(pl.exclude(pl.String())).to_pandas()
        )
        df_imputed.to_csv("imputed_cache.csv", index=False)
    else:
        print("b")
        df_imputed = pl.read_csv("imputed_cache.csv").to_pandas()
    return df_imputed, missForestImputer


@app.cell
def __(df_consistent, pl, shapiro):
    pl.DataFrame(
        [
            list(shapiro(df_consistent["nutrient_k_ppm"])),
            list(shapiro(df_consistent["nutrient_p_ppm"])),
            list(shapiro(df_consistent["nutrient_n_ppm"])),
            list(shapiro(df_consistent["humidity_percent"])),
            list(shapiro(df_consistent["light_intensity_lux"])),
        ]
    ).transpose(column_names=["statistic", "p value"]).hstack(
        pl.DataFrame(
            [
                "nutrient_k_ppm",
                "nutrient_p_ppm",
                "nutrient_n_ppm",
                "humidity_percent",
                "light_intensity_lux",
            ],
            schema=["feature"],
        )
    ).select("feature", "statistic", "p value")
    return


@app.cell
def __(mo):
    mo.md(r"""## Test of homogenity of variance""")
    return


@app.cell
def __(bartlett, df_consistent, df_imputed, pl):
    def bertlett_helper(feature_name, df_consistent):
        l = []
        for classi in df_consistent["plant_type_stage"].unique().to_list():
            l.append(
                df_consistent.filter(pl.col("plant_type_stage") == classi)[
                    feature_name
                ].to_list()
            )
        return l


    _d = pl.from_pandas(df_imputed).hstack(
        df_consistent.select("plant_type_stage")
    )

    pl.DataFrame(
        (
            list(bartlett(*bertlett_helper("nutrient_k_ppm", _d))),
            list(bartlett(*bertlett_helper("nutrient_p_ppm", _d))),
            list(bartlett(*bertlett_helper("nutrient_n_ppm", _d))),
            list(bartlett(*bertlett_helper("humidity_percent", _d))),
            list(bartlett(*bertlett_helper("light_intensity_lux", _d))),
        )
    ).transpose(column_names=["statistic", "p value"]).hstack(
        pl.DataFrame(
            [
                "nutrient_k_ppm",
                "nutrient_p_ppm",
                "nutrient_n_ppm",
                "humidity_percent",
                "light_intensity_lux",
            ],
            schema=["feature"],
        )
    ).select("feature", "statistic", "p value")
    return (bertlett_helper,)


@app.cell
def __(mo):
    mo.md(r"""normality and homogeneity of variance are held""")
    return


@app.cell
def __(df_consistent, df_imputed, pl, ttest_1samp):
    _d = pl.from_pandas(df_imputed).hstack(
        df_consistent.select("plant_type_stage")
    )

    pl.DataFrame(
        (
            list(ttest_1samp(_d["nutrient_k_ppm"], _d["nutrient_k_ppm"].mean())),
            list(ttest_1samp(_d["nutrient_p_ppm"], _d["nutrient_p_ppm"].mean())),
            list(ttest_1samp(_d["nutrient_n_ppm"], _d["nutrient_n_ppm"].mean())),
            list(
                ttest_1samp(_d["humidity_percent"], _d["humidity_percent"].mean())
            ),
            list(
                ttest_1samp(
                    _d["light_intensity_lux"], _d["light_intensity_lux"].mean()
                )
            ),
        )
    ).transpose(column_names=["statistic", "p value"]).hstack(
        pl.DataFrame(
            [
                "nutrient_k_ppm",
                "nutrient_p_ppm",
                "nutrient_n_ppm",
                "humidity_percent",
                "light_intensity_lux",
            ],
            schema=["feature"],
        )
    ).select("feature", "statistic", "p value")
    return


@app.cell
def __(mo):
    mo.md(r"""##Effect Size (I failed in this one)""")
    return


@app.cell
def __(association, df_imputed, pl):
    _d = pl.from_pandas(df_imputed)
    _d = _d.with_columns(pl.all().cast(pl.Int64))
    pl.DataFrame(
        (
            list(
                association(
                    _d["nutrient_k_ppm"],
                )
            ),
            list(
                association(
                    _d["nutrient_p_ppm"],
                )
            ),
            list(
                association(
                    _d["nutrient_n_ppm"],
                )
            ),
            list(
                association(
                    _d["humidity_percent"],
                )
            ),
            list(
                association(
                    _d["light_intensity_lux"],
                )
            ),
        )
    ).transpose(column_names=["statistic", "p value"]).hstack(
        pl.DataFrame(
            [
                "nutrient_k_ppm",
                "nutrient_p_ppm",
                "nutrient_n_ppm",
                "humidity_percent",
                "light_intensity_lux",
            ],
            schema=["feature"],
        )
    ).select("feature", "statistic", "p value")
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""# Regression  hypothesis testing using OLS linear regression against target""")
    return


@app.cell
def __(df_consistent, df_imputed, pl, sm):
    _df = df_consistent.filter(~pl.col("temperature_celsius").is_nan())
    _y = _df["temperature_celsius"].to_pandas()
    _x = (
        pl.from_pandas(df_imputed)
        .hstack(
            df_consistent.select("temperature_celsius").rename(
                {"temperature_celsius": "temperature_celsius_2"}
            )
        )
        .filter(pl.col("temperature_celsius_2").is_not_nan())
        .drop(
            "temperature_celsius_2",
        )
    ).to_pandas()
    print(_x.columns)
    model = sm.OLS(_y, _x).fit()
    model.summary()
    return (model,)


@app.cell
def __(mo):
    mo.md(
        r"""
        temperature_celsius	1.0000	1.44e-17	6.94e+16	0.000	1.000	1.000  
        humidity_percent	-5.447e-16	1.92e-17	-28.384	0.000	-5.82e-16	-5.07e-16  
        light_intensity_lux	-9.842e-17	6.05e-19	-162.747	0.000	-9.96e-17	-9.72e-17  
        co2_ppm	2.374e-17	8.19e-19	28.995	0.000	2.21e-17	2.53e-17   
        ec_dsm	-6.384e-16	2.66e-16	-2.396	0.017	-1.16e-15	-1.16e-16  
        o2_ppm	1.921e-16	1e-16	1.919	0.055	-4.07e-18	3.88e-16  
        nutrient_n_ppm	-1.392e-16	4.19e-18	-33.203	0.000	-1.47e-16	-1.31e-16  
        nutrient_p_ppm	1.078e-15	1.15e-17	93.639	0.000	1.06e-15	1.1e-15  
        nutrient_k_ppm	-2.559e-17	2.89e-18	-8.853	0.000	-3.13e-17	-1.99e-17  
        ph	-1.11e-16	2.61e-16	-0.426	0.670	-6.22e-16	4e-16  
        water_level_mm	3.123e-17	1.98e-17	1.576	0.115	-7.61e-18	7.01e-17  
        plant_type_changed	-5.829e-16	2.8e-16	-2.078	0.038	-1.13e-15	-3.32e-17  
        plant_stage_coded	9.048e-17	1.99e-16	0.455	0.649	-2.99e-16	4.8e-16  
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # feature v feature hypothesis testing
        ## nutrient n vs nutrient k
        """
    )
    return


@app.cell
def __(df_imputed, pl, sm):
    _model = sm.OLS(
        pl.from_pandas(df_imputed).select("nutrient_n_ppm").to_pandas(),
        df_imputed["nutrient_k_ppm"],
    ).fit()
    _model.summary().tables[1]
    return


@app.cell
def __(mo):
    mo.md(r"""## nutrient n vs nutrient p""")
    return


@app.cell
def __(df_imputed, pl, sm):
    _model = sm.OLS(
        pl.from_pandas(df_imputed).select("nutrient_n_ppm").to_pandas(),
        df_imputed["nutrient_p_ppm"],
    ).fit()
    _model.summary().tables[1]
    return


@app.cell
def __(mo):
    mo.md(r"""## nutrient p vs nutrient k""")
    return


@app.cell
def __(df_imputed, pl, sm):
    _model = sm.OLS(
        pl.from_pandas(df_imputed).select("nutrient_p_ppm").to_pandas(),
        df_imputed["nutrient_k_ppm"],
    ).fit()
    _model.summary().tables[1]
    return


@app.cell
def __(mo):
    mo.md("""## nutrient n vs light intensity lux""")
    return


@app.cell
def __(df_imputed, np, smf):
    _df = df_imputed[["nutrient_n_ppm", "light_intensity_lux"]]
    _df.columns = ["y", "x"]


    def poly(x, degree):
        return np.vander(x, degree + 1, increasing=True)[:, 1:]


    results = smf.ols(formula="y ~ poly(x, 2)", data=_df).fit()
    results.summary().tables[1]
    return poly, results


@app.cell
def __(mo):
    mo.md("""## nutrient p vs light intensity lux""")
    return


@app.cell
def __(df_imputed, results, smf):
    results
    _df = df_imputed[["nutrient_p_ppm", "light_intensity_lux"]]
    _df.columns = ["y", "x"]


    _results = smf.ols(formula="y ~ poly(x, 2)", data=_df).fit()
    _results.summary().tables[1]
    return


@app.cell
def __(mo):
    mo.md("""## nutrient k vs light intensity lux""")
    return


@app.cell
def __(df_imputed, smf):
    _df = df_imputed[["nutrient_k_ppm", "light_intensity_lux"]]
    _df.columns = ["y", "x"]


    _results = smf.ols(formula="y ~ poly(x, 2)", data=_df).fit()
    _results.summary().tables[1]
    return


@app.cell
def __(mo):
    mo.md(r"""## plant_stage_coded v nutrient_k_ppm""")
    return


@app.cell
def __(df_imputed, pl, sm):
    _model = sm.OLS(
        pl.from_pandas(df_imputed).select("plant_stage_coded").to_pandas(),
        df_imputed["nutrient_k_ppm"],
    ).fit()
    _model.summary().tables[1]
    return


@app.cell
def __(mo):
    mo.md(r"""## plant_stage_coded v nutrient_n_ppm""")
    return


@app.cell
def __(df_imputed, pl, sm):
    _model = sm.OLS(
        pl.from_pandas(df_imputed).select("plant_stage_coded").to_pandas(),
        df_imputed["nutrient_n_ppm"],
    ).fit()
    _model.summary().tables[1]
    return


@app.cell
def __(mo):
    mo.md(r"""## plant_stage_coded v nutrient_p_ppm""")
    return


@app.cell
def __(df_imputed, pl, sm):
    _model = sm.OLS(
        pl.from_pandas(df_imputed).select("plant_stage_coded").to_pandas(),
        df_imputed["nutrient_p_ppm"],
    ).fit()
    _model.summary().tables[1]
    return


@app.cell
def __(mo):
    mo.md(r"""## the nutrients are really significantly  related to eaach other and these other features like light intensity and plant_stage_coded. nutrients p and k can be removed as discussed in the visualisation sections""")
    return


if __name__ == "__main__":
    app.run()
