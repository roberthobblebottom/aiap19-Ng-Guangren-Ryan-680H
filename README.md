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
  
- d) For both `plant_type_stage` Classification and `temperature_celsius` Regression Tasks:  
  1. connection to database  
  2. rename columns and make feature values consistent  
  3. remove non important features as discussed in eda.
  4. init pipeline  
    1. `outliersRemover()` - Removes outliers
    2. `ColumnsTransformerForOneHotEncoding()`  one hot encoding on selected String features. for `temperature_celsius` Regression Task only
    3. `SimpleImputer()` Median Strategy 
    4. Either `RandomForestClassifier` or `RandomForestRegressor` for `Plant_Type_Stage` and `Temperature` respectively  
  5. train(): `train_test_split()`   
  6. train(): `RandomSearchCV()` with 5 fold cross validation    

        step 4 and 5 rationale: 3 way hold out with cross validation it is used because  
        unlike 2 way hold out, the test error less depended on samples use  
        and may overestimate test error   
        these split is used in such a way that data leakage isn't present.  
        Data Leakage will give a overestimated test error.  
          
  7. `evaluate()`: the `pipeline.predict()` and `pipeline.predict_proba()` and respective metrics are calculated on respective model tasks   
  
- e) the nutrients are staticially significantly  related to each other and these other features like `light_intensity_lux`, `humidity_percent` and `plant_stage_coded`. (alpha = 0.05)   
  
the p values for `temperature_celsius` regression task compared to the target (temperature) are significant, nutrients p and k can be removed as discussed in the visualisation sections. but for classfication task since k is the only one significant of all the nutrients, p and n will be removed  
  
- f)

  
| feature         | Processes         |   
| :----------- | :--------------- |  
| temperature_celsius| used as target for Temperature.py pipeline, removedd those that are Nan. for Plant_Type_Stage regression pipeline, temperature below 0 is removed|  
| plant_type_stage | created using plant_type and plant_stage. used as target for Plant_Type_Stage.py pipeline. one hot encoded for `temperature_celsius` regression task|  
| plant_type | removed in `plant_type_stage` classification task to prevent data leakage.set to lower case and space character replaced with underscores. |  
|plant_stage|removed in `plant_type_stage` classification task to prevent data leakage. one hot encoded in the `temperature_celsius` regression task|
| humidity_percent    | Remove any outliers beyond  the range of 0 to 100   |   
| light_intensity_lux | removed any negatives   | 
| co2_ppm    | removed any negatives | 
| ec_dsm    | removed any negatives   | 
| o2_ppm    | set o2 for frutiing vegetables and herbs into certain range to remove outliers as mentioned in the eda. removed any negatives  | 
| nutrient_n_ppm    | removed " ppm" substring from the strings if such substring exist. casted to float. removed any negatives   | 
| ph    |maded sure values are within range of 0 and 14. casted to integer| 
| water_level_mm    |    | 
| nutrient_k_ppm    |  removed " ppm" substring from the strings if such substring exist. casted to float.removed any negatives   | 
| nutrient_p_ppm    | removed " ppm" substring from the strings if such substring exist. casted to float. removed any negatives  | 
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
  - `temperature_celsius` Regression Task:    
    - Mean Absolute Error: Rombust to outliers,Interpretability  
    -Root Mean Squared Error: More interpretable than mean squared error, Less sensitive to large errors. The RMSE tells us how well a regression model can predict the value of the response variable in absolute terms while R2 tells us how well a model can predict the value of the response variable in percentage terms.
    - R2 score: scale indepedent
    ease of interpretability
  
  - `plant_type_stage` Classification Task:  
    - Accuracy: Useful for looking at all classes.  
    - F1: Balances between precision and recall; the harmonic mean between the two.  
            Useful if the aim is to look at the positives classes only  
    - AUC-ROC: provides insights into the performance of the model  
    - a balance look between type 1 and type 2 errors would be a good idea for plant_type_stage so accuracy will be the priority metrics to look at.
  
- i) EDA is initially done on marimo notebook and exported to ipynb format. You may also look at the marimo notebook output via opening marimo_notebooks/eda.html 

unable to make the pipeline more syncronised as mentioned in my aiap16 assessment feedback. unable to improve temperature metrics.  
  
