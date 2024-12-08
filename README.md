a) Ng Guangren, Ryan ryan.ng@protonmail.com

b) folder structure: 
eda.py
requirements.txt  
eda.ipynb
__marimo__ TODO
README.md
run.sh
src
- Temperature.py
- Plant_Type_Stage.py

c) in command line, type in:
cd <TO-MY-SUBMISSION-FOLDER>
pip install -r requirements.txt
Bash run.sh # to run both Temperature and Plant_Type_Stage pipelines

d) For both Classification and Regression Tasks:
1 connection to database
2 rename columns and make feature values consistent
3 init pipeline
        NonImportantFeaturesRemover() As discussed in EDA, removing the non important features
        outliersFremover()
        ColumnsTransformerForOneHotEncoding()
        SimpleImputer() Median Strategy
        Either RandomForestClassifier or RandomForestRegressor for Plant_Type_Stage and Temperature respectively
4 train(): train test split
5 train(): RandomSearchCV() with 5 fold cross validation
6 evaluate(): the pipeline.predict() and pipeline.predict_proba() and respective metrics are calculated on respective model tasks 

e) the nutrients are staticially significantly  related to each other and these other features like light_intensity_lux, humidity_percent and plant_stage_coded. (alpha = 0.05) 

the p values for regression task compared to the target  (temperature) are significant, nutrients p and k can be removed as discussed in the visualisation sections. but for classfication task since k is the only one significant of all the nutrients, p and n will be removed

f) Describe how the features in the dataset are processed (summarised in a table).
TODO

g) 
Both tasks uses RandomForests from sklearn:
RandomForestClassifier
RandomForestRegressor

Rationale:
High acurracy
robust to outliers and noise (but I will still remove the outliers as found in the eda. probably has little effect including the outliers remover in the pipeline)
handles combination of numerical and categorial.
non parametic nture where it does not assume things about the distribution or correlations between x and y.
but SimpleImputer requires all to be numerical so 
less likely to overfit than gradientboosting models and some linear regression models.

but the downside probabilty less accurate than gradientboosting models

h) Rational for metrics used:

Regression Task: 
        Mean Absolute Error: Rombust to outliers,Interpretability
        Root Mean Squared Error: More interpretable than mean squared error, Less sensitive to large errors

Classification Task:
        Accuracy: Useful for looking at all classes.
        F1: Balances between precision and recall; the harmonic mean between the two.
            Useful if the aim is to look at the positives classes only
        AUC-ROC: provides insights into the performance of the model

i) EDA is initially done on marimo notebook and exported to ipynb format.
