# bulldozer-price-prediction (Work in Progress)

<p align="center">
<img src="https://user-images.githubusercontent.com/86231828/125293986-442f0180-e367-11eb-9a0e-024d517a6d36.jpg" height=250 width=auto>  
</p>

## An end-to-end data analysis and machine learning workflow to predict bulldozer sales prices.

This is a repository to store a exploratory data analysis and machine learning workflow for Kaggle's ["Blue book for bulldozers" challenge](https://www.kaggle.com/c/bluebook-for-bulldozers/), which aims to accurately predict a bulldozer's auction price given its features and sale date. A Jupyter notebook detailing the problem to be solved, exploratory data analysis, and supervised training is uploaded. This time, the final trained model is not provided because it exceeds Github's upload limit.

### Summary

The data provided had to be preprocessed before it could be passed to a machine learning model. I had to do this process in 3 steps:
* Parse the sale date entries to enrich the data
* Convert string datatypes into Pandas categories
* Impute missing data.
  * Missing data was filled in with the median of all entries in a feature.
  * 1 was added to categorical data, as Pandas gives a code of -1 to missing data by default.  

Before preprocessing, I made sure to **keep training and validation data separate** due to the median imputation method for numerical data. If the data was imputed before splitting, there would be a mix in the training and validation data which may cause the machine learning algorith to discern spurious patterns which would otherwise hamper its ability to generalize future data.

The model I ended up choosing was a `RandomForestRegressor` taken from the [Scikit-Learn package](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) and tuning the hyperparameters via 5-fold cross-validation on a subset of 10,000 pieces of data, given that the full training set was a little more then 400,000 in quantity. Despite this, training the model on the full spread of data shows that the model has a markedly better performance on the training data rather than the validation data, see the coefficient of determination and the root mean squared log error below, suggesting an overfitted model.

<p align="center">
<img src="https://user-images.githubusercontent.com/86231828/125293048-63795f00-e366-11eb-8f0a-202f1bdce658.jpg" height=400 width=auto>  
</p>

For completeness, I provide the feature importance of the model below:

<p align="center">
<img src="https://user-images.githubusercontent.com/86231828/125294167-66288400-e367-11eb-9809-f8b0388a5d57.jpg" height=450, width=auto>
</p>

There is much more that could be done to squeeze more performance out of the data provided, namely, by using a different regression model, such as CatBoost, or devising alternative methods to preprocess the data for machine learning. As such, this is a continuing work.
