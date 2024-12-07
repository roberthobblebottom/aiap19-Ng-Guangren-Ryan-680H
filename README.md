a) Ng Guangren, Ryan ryan.ng@protonmail.com

b) folder structure: TODO



c) in command line, do `Bash run.sh` to run both Temperature and Plant_Type_Stage pipelines


d) Description of logical steps/flow of the pipeline. If you find it useful, please feel free to
include suitable visualisation aids (eg, flow charts) within the README.
TODO

e) Overview of key findings from the EDA conducted in Task 1 and the choices made in the
pipeline based on these findings, particularly any feature engineering. Please keep the
details of the EDA in the `.ipynb`. The information in the `README.md` should be a quick
summary of the details from `.ipynb`.

## the nutrients are staticially significantly  related to each other and these other features like light_intensity_lux, humidity_percent and plant_stage_coded. (alpha = 0.05) 

##the p values for regression task compared to the target  (temperature) are significant, nutrients p and k can be removed as discussed in the visualisation sections. but for classfication task since k is the only one significant of all the nutrients, p and n will be removed

f) Describe how the features in the dataset are processed (summarised in a table).
TODO


g Explanation of your choice of models for each machine learning task.
TODO



h) Rational for metrics used:

Regression Task: 
        Mean Absolute Error: Rombust to outliers,Interpretability
        Root Mean Squared Error: More interpretable than mean squared error, Less sensitive to large errors

Classification Task:
        Accuracy: Useful for looking at all classes.
        F1: Balances between precision and recall; the harmonic mean between the two.
            Useful if the aim is to look at the positives classes only
        AUC-ROC: prvides insights into the performance of the model

