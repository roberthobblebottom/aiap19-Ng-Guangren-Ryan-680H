- a) Ng Guangren, Ryan ryan.ng@protonmail.com  
  
- b) folder structure:  
requirements.txt    
eda.ipynb  
marimo_notebooks/   
-- eda.py  
-- eda.html  
README.md  
run.sh  
src/  
-- Temperature.py  
-- Plant_Type_Stage.py  
  
- c) in command line, type in:  
`cd path/to/aiap19-Ng-Guangren-Ryan-680H`  
`pip install -r requirements.txt`  
`bash run.sh` # to run both `Temperature` and `Plant_Type_Stage` pipelines  `\
  
- d) For both Classification and Regression Tasks:  
  1. connection to database  
  2. rename columns and make feature values consistent  
  3. init pipeline  
    1. `NonImportantFeaturesRemover()` As discussed in EDA, removing the non important features  
    2. `outliersRemover()` - Remkoves outliers
    3. `ColumnsTransformerForOneHotEncoding()`  
    4. `SimpleImputer()` Median Strategy 
    5. Either `RandomForestClassifier` or `RandomForestRegressor` for `Plant_Type_Stage` and `Temperature` respectively  
  4. train(): `train_test_split()`   
  5. train(): `RandomSearchCV()` with 5 fold cross validation    

step 4 and 5 rationale: 3 way hold out with cross validation it is used because  
unlike 2 way hold out, the test error less depended on samples use  
and may overestimate test error   
these split is used in such a way that data leakage isn't present.  
Data Leakage will give a overestimated test error.  
          
6. `evaluate()`: the `pipeline.predict()` and `pipeline.predict_proba()` and respective metrics are calculated on respective model tasks   
  
- e) the nutrients are staticially significantly  related to each other and these other features like `light_intensity_lux`, `humidity_percent` and `plant_stage_coded`. (alpha = 0.05)   
  
the p values for regression task compared to the target (temperature) are significant, nutrients p and k can be removed as discussed in the visualisation sections. but for classfication task since k is the only one significant of all the nutrients, p and n will be removed  
  
- f) Describe how the features in the dataset are processed (summarised in a table).  
all categorial features are set to lower case and space character replaced with underscores.
| feature         | Processes         | 
| :----------- | :--------------- |
| temperature_celsius| used as target for Temperature.py pipeline|
| plant_type_stage | created using plant_type and plant_stage. used as target for Plant_Type_Stage.py pipeline|
| plant_type | set to lower case and space character replaced with underscores. |
| humidity_percent    | Even more text   | 
| light_intensity_lux | Even more text   | 
| co2_ppm    | Even more text   | 
| ec_dsm    | Even more text   | 
| o2_ppm    | Even more text   | 
| nutrient_n_ppm    | removed " ppm" substring from the strings if such substring exist. casted to float   | 
| ph    | Even more text   | 
| water_level_mm    | Even more text   | 
| nutrient_k_ppm    |  removed " ppm" substring from the strings if such substring exist. casted to float   | 
| nutrient_p_ppm    | removed " ppm" substring from the strings if such substring exist. casted to float  | 
| plant_stage_coded    | removed from both pipelines   | 
| previous_cycle_plant_type    | removed from both pipelines   | 
| location    | removed from both pipelines   | 
 
- g) Both tasks uses RandomForests from sklearn:  
RandomForestClassifier for Plant_Type_Stage task and RandomForestRegressor for Temperature task  
  
Rationale:  
High accuracy  
robust to outliers and noise (but outliers are still removed as found in the eda. probably has little effect including the outliers remover in the pipeline, likely more useful for gradientboosting models)  
~~handles combination of numerical and categorial.~~  (upon testing RandomForestRegressor and RandomForestClassifer requires all features to be normial)
non parametic nture where it does not assume things about the distribution or correlations between x and y.  
but `SimpleImputer` and the Random Forest models requires all to be numerical   
less likely to overfit than gradientboosting models and some linear regression models.  
  
but the downside probably less accurate than gradientboosting models  
  
- h) Rational for metrics used:  
  - Regression Task:    
    - Mean Absolute Error: Rombust to outliers,Interpretability  
    -Root Mean Squared Error: More interpretable than mean squared error, Less sensitive to large errors  
  
  - Classification Task:  
    - Accuracy: Useful for looking at all classes.  
    - F1: Balances between precision and recall; the harmonic mean between the two.  
            Useful if the aim is to look at the positives classes only  
    - AUC-ROC: provides insights into the performance of the model  
  
- i) EDA is initially done on marimo notebook and exported to ipynb format. You may also look at the marimo notebook output via opening marimo_notebooks/eda.html  
