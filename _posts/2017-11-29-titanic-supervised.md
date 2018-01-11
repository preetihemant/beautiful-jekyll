---
layout: post
title: Supervised Machine Learning
subtitle: Titanic disaster 
image: /img/Titanic_Survival_Decison_Tree_SVG.png
bigimg: /img/Titanic_Survival_Decison_Tree_SVG.png
---

<p> Titanic ship that sailed on 10th April 1912 and sank 5 days later had 2224 passengers aboard. Despite all the safety measures on 
the ship, 1502 passengers died leaving only 722 survivors. Our sample is a set of the most complete records from the original set.
We will use this data set to run through all the steps in a typical data analysis exercise </p>

1. <b> Data exploration and visualization </b> - This involves data cleaning, outlier removal and inspecting and
visualizing data for any anomalies,correlations and handling of missing data.
2. <b> Feature engineering </b>
  * Feature scaling - 2 or more features that have different ranges can be compared and used in 
analysis to give better results if transformed to the same scale. There are different types of scalers: minmax scaler, mean subtraction, standard scaler.
  * Feature creation - We may want to create new features from the existing feature to include more information for our ML model. For this we may combine two or more features and create a new one.
  * Feature representation - Ml models require that input data be in numerical format. Many of the features are typically categorical or non-numerical. In such cases, we will need to apply some kind of transformation like vectorization or discretization to make these values ML model compatible.
  * Feature selection - Not all features have the same amount of information. As the number of features increases, the ML process increases in time and complexity.
Many features also dont carry any insights or are redundant and can be eliminated. Data visualization helps to an extent in selecting the best features. There
also are mathematical ways of doing this - Recursive Feature elimination (RFE) , k-best percentile.
  * Feature transforms - Correlated features tend to vary together and using only one of them is sufficient in ML models. To understand the degree
of correlation we use PCA - principle component analysis. 
3. <b> Machine learning models </b>
  * Supervised or unsupervised learning - Depending on what kind of data we have, an appropriate model can be selected. For supervised data that is
ordered or continuous we often use linear regression or decision tree regression. For supervised data that is non-ordered or discreet, we can 
use Naive-bayes, SVM or KNN. Popular unsupervised learning models are k-means clustering, spectral clustering.
  * Validation - Once we have trained our model, we validate its working on a set of test samples.
  * Tuning of algorithm - Various parameters in the models can be tuned to improve performance on test data.
  * Performance evaluation - Typical metrics used are accuracy score, precision and recall. 

<p> Let us now apply this process to the titanic survival data set. The data has already been split into train and test. We will work on the train data and any modifications we make will also be applied the test set (holdout data). </p>
<p> Let us first inspect the training data </p>

```python
import pandas as pd

filename_train = ("train.csv")
titanic_df = pd.read_csv(filename_train)

filename_test = ("test.csv")
holdout_df=pd.read_csv(filename_test)

print titanic_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
None
```
```python
print titanic_df.describe()
       PassengerId    Survived      Pclass         Age       SibSp  
count   891.000000  891.000000  891.000000  714.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118    0.523008
std     257.353842    0.486592    0.836071   14.526497    1.102743
min       1.000000    0.000000    1.000000    0.420000    0.000000
25%     223.500000    0.000000    2.000000   20.125000    0.000000
50%     446.000000    0.000000    3.000000   28.000000    0.000000
75%     668.500000    1.000000    3.000000   38.000000    1.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000

  Parch        Fare
count  891.000000  891.000000
mean     0.381594   32.204208
std      0.806057   49.693429
min      0.000000    0.000000
25%      0.000000    7.910400
50%      0.000000   14.454200
75%      0.000000   31.000000
max      6.000000  512.329200
```
<p> </p>
#### Handling of missing data
* Age - We will fill up the missing values with a negative number and treat it as missing when categorized
* Embarked - We will fill the missing values with the most common station from where people embarked which happens to be"S"
* Fare - There are no missing values in the training set howeverthe holdout data or the test set has some. We will fill them up by the average fare from the training data

```python
titanic_df["Age"] = titanic_df["Age"].fillna(-0.5)
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

holdout_df["Age"] = holdout_df["Age"].fillna(-0.5)
holdout_df["Embarked"] = holdout_df["Embarked"].fillna("S")

# Missing fare in holdout data, no missing fare in training data
holdout_df["Fare"] = holdout_df["Fare"].fillna(titanic_df["Fare"].mean())
```
<p> </p>
#### Feature creation
Let us create a new feature called Family_size which will combine two features together - SibSp and Parch. This feature is simply a sum of SibSp and Parch.
```python
titanic_df["Family_size"]=titanic_df["SibSp"] + titanic_df["Parch"]
holdout_df["Family_size"]=holdout_df["SibSp"] + holdout_df["Parch"]
```
<p> </p>

#### Feature scaling
Range of SibSp is [0, 8], Parch is [0, 6] and Fare is [0, 512]. Let us map these 3 features and the new one family size to the same range using the minmax scaler

```python
from sklearn.preprocessing import minmax_scale
columns=["SibSp","Parch","Fare","Family_size"]
for col in columns:
    titanic_df[col+"_scaled"] = minmax_scale(titanic_df[col])
    holdout_df[col+"_scaled"] = minmax_scale(holdout_df[col])
```
<p> </p>

#### Feature binning
Age and fare are a continuous series. Since other parameters like Pclass, Sex are categorical, these two can be converted into categories. To do this, we can use cut function in pandas series
```python
cut_points_age=[-1,0,5,12,18,35,60,100]
label_names_age=["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
titanic_df["Age_categories"] = pd.cut(titanic_df["Age"],cut_points_age,labels=label_names_age)
holdout_df["Age_categories"] = pd.cut(holdout_df["Age"],cut_points_age,labels=label_names_age)

cut_points_fare = [0,12,50,100,1000]
label_names_fare = ["0-12","12-50","50-100","100+"]
titanic_df["Fare_categories"]=pd.cut(titanic_df["Fare"],cut_points_fare,labels=label_names_fare)
holdout_df["Fare_categories"]=pd.cut(holdout_df["Fare"],cut_points_fare,labels=label_names_fare)
```
<p> </p>

#### Feature extraction
A closer look at the name and cabin type features shows us that they too contain information relevant to survival data. Names with Sir, Countess, Lady and so on imply royalty and therefore a first class ticket and hence a higher probability of survival. Similary, cabin type too conveys survival probability. Let us now extract information from these features
```python
titles = {
    "Mr" :         "Mr",
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Master" :     "Master",
    "Mlle":        "Miss",
    "Miss" :       "Miss",
    "Capt":        "Officer",
    "Col":         "Officer",
    "Major":       "Officer",
    "Dr":          "Officer",
    "Rev":         "Officer",
    "Jonkheer":    "Royalty",
    "Don":         "Royalty",
    "Sir" :        "Royalty",
    "Countess":    "Royalty",
    "Dona":        "Royalty",
    "Lady" :       "Royalty"
}

extracted_titles = titanic_df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
titanic_df["Title"] = extracted_titles.map(titles)

extracted_titles = holdout_df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
holdout_df["Title"] = extracted_titles.map(titles)

titanic_df["Cabin_type"] = titanic_df["Cabin"].str[0]
titanic_df["Cabin_type"]=titanic_df["Cabin_type"].fillna("Unknown")

holdout_df["Cabin_type"] = holdout_df["Cabin"].str[0]
holdout_df["Cabin_type"]=holdout_df["Cabin_type"].fillna("Unknown")
```
<p> </p>

#### Feature representation
To prepare our columns for machine learning, most algorithms dont userstand text labels and these values need to be converted into numbers
```python
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df
    
for col in ["Title","Cabin_type","Fare_categories","Embarked","Age_categories","Pclass","Sex"]:
    titanic_df = create_dummies(titanic_df,col)
    holdout_df = create_dummies(holdout_df,col)
```


#### Picking the best performing features
At the end of all the steps of feature prepartion, we now have over 30 features. Surely some of them must be correlated, some other irrelevant. We could always feed all of them into our ML model, but that will increase training time and complexity. Let us instead cut this list down to the best performing features.

* #### Through elimination of correlated features
As a first step we can eliminate features that are highly correlated since they would contain repeated information.

```python
import seaborn as sns

def plot_correlation_heatmap(df):
    corr = df.corr()
    
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)


    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.2, cbar_kws={"shrink": .2})
    plt.show()

columns = ['Family_size_scaled','Fare_scaled','Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty', 'Cabin_type_A', 'Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G', 'Cabin_type_T', 'Cabin_type_Unknown', 'Fare_categories_0-12', 'Fare_categories_100+', 'Fare_categories_12-50', 'Fare_categories_50-100', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age_categories_Adult', 'Age_categories_Child', 'Age_categories_Infant', 'Age_categories_Missing', 'Age_categories_Senior', 'Age_categories_Teenager', 'Age_categories_Young Adult', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male']

plot_correlation_heatmap(titanic_df[columns])
```
![heat map](/img/heat_diagram.png)

<p> We can see that there is a high correlation between Sex_female/Sex_male and Title_Miss/Title_Mr/Title_Mrs. We will remove the columns Sex_female and Sex_male since the title data may be more nuanced.
Apart from that, we should remove one of each of our dummy variables to reduce the collinearity in each. We'll remove: </p>

* Pclass_2
* Age_categories_Teenager
* Fare_categories_12-50
* Title_Master
* Cabin_type_A

Our list of uncorrelated features now is 
```python
columns_uncorr=['Family_size_scaled','Fare_scaled', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty', 'Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G', 'Cabin_type_T', 'Cabin_type_Unknown', 'Fare_categories_0-12', 'Fare_categories_100+', 'Fare_categories_50-100', 'Embarked_Q', 'Embarked_S', 'Age_categories_Adult', 'Age_categories_Child', 'Age_categories_Infant', 'Age_categories_Missing', 'Age_categories_Senior', 'Age_categories_Teenager', 'Age_categories_Young Adult', 'Pclass_1',  'Pclass_3', ]
```
&nbsp;

* #### Through coefficients of features
<p> We can also eliminate features by looking at the coefficients of each feature. Once the model is trained, we can access this attirbute. The coef() method returns a NumPy array of coefficients, in the same order as the features that were used to fit the model. We can then select the top 10 best features to train our model. </p>

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

columns = ['Family_size_scaled','Fare_scaled','Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty', 'Cabin_type_A', 'Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G', 'Cabin_type_T', 'Cabin_type_Unknown', 'Fare_categories_0-12', 'Fare_categories_100+', 'Fare_categories_12-50', 'Fare_categories_50-100', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age_categories_Adult', 'Age_categories_Child', 'Age_categories_Infant', 'Age_categories_Missing', 'Age_categories_Senior', 'Age_categories_Teenager', 'Age_categories_Young Adult', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male']

clf = LogisticRegression()
clf.fit(titanic_df[columns],titanic_df["Survived"])
coefficients = clf.coef_
feature_importance = pd.Series(coefficients[0],index=columns)

ordered_feature_importance = feature_importance.abs().sort_values()
ordered_feature_importance.plot.barh()
plt.show()
```
![Feature coeff](/img/titanic_features.png)

* #### Through RFECV
The process described above requires manual selection of features. To automate picking the best features, there is RFECV - Recursive Feature Elimination with Cross Validation. Here I have selected logistic regression model and the previously determined uncorrelated features to run RFECV on.

```python
from sklearn.feature_selection import RFECV
lr = LogisticRegression()
selector=RFECV(lr,cv=10)
selector.fit(titanic_df[columns_uncorr],titanic_df["Survived"])
optimized_columns=titanic_df[columns_uncorr].columns[selector.support_]
```
&nbsp;

#### Accuracy from the reduced set of features
We can compare the accuracy obtained with the two approaches of selecting the best features - 1) feature coefficients 2) RFECV on uncorrelated features
```python
from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(clf,titanic_df[columns_best_perf],titanic_df["Survived"],cv=10)
accuracy = np.mean(scores)
print (accuracy)
```
Feature coefficients gives us an average accuracy of 81.49%. 

```python
from sklearn.feature_selection import RFECV
lr = LogisticRegression()
selector=RFECV(lr,cv=10)
selector.fit(titanic_df[columns_uncorr],titanic_df["Survived"])
optimized_columns=titanic_df[columns_uncorr].columns[selector.support_]

scores = cross_val_score(lr,titanic_df[optimized_columns],titanic_df["Survived"],cv=10)
accuracy = np.mean(scores)
print (accuracy)
```
RFECV method gives an accuracy of 82.93%. This method definitely seems to give us better performance.

#### Randomforest classifier
Let us now try a different model - RandomForest classifier. Without any fine tuning of the model, let us evaluate the performance using the features optimized through RFECV .

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=1)
scores = cross_val_score(clf,titanic_df[optimized_columns],titanic_df["Survived"],cv = 10)
accuracy_rf = np.mean(scores)
print ("random forest model",accuracy_rf)
```
We get an accuracy of 83.16% which is an improvement over the logistic regression model. Can we further increase the performance by changing model parameters? Let us try!

We will use gridsearchcv that performs an exhaustive search for the specified parameter values for an estimator. For the randomforest classifier, we will try getting the best performance over the parameters below.

```python
hyperparameters = {"criterion": ["entropy", "gini"],
                   "max_depth": [5, 10],
                   "max_features": ["log2", "sqrt"],
                   "min_samples_leaf": [1, 5],
                   "min_samples_split": [3, 5],
                   "n_estimators": [6, 9]
}
```

```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(clf,param_grid=hyperparameters,cv=10)
grid.fit(titanic_df[optimized_columns],titanic_df["Survived"])
best_params = grid.best_params_
best_score = grid.best_score_
print best_score
```

We get a slight improvement in the accuracy, it is now 83.39%. Is the improvement worth the extra time and computation that gridsearchcv does? It depends on our application. Some require a very high accuracy and even a slight improvement would be desirable.

In conclusion, randomforest classifier with the optimized columns from RFECV method gave us the best performance. 

#### Fine tuning our algorithm
<p> So far we validated our models using a small subset of our original data that we set aside for testing. Let us now apply our best model to the holdout data and check how well the algorithm does. </p>

```python
best_rf = grid.best_estimator_   #select the best RF classifier with optimal hyperparameters
holdout_predictions = best_rf.predict(holdout_df[optimized_columns])
```
#### And the kaggle score is ...
We are using kaggle score to determine how good we did in comparison to others. A kaggle submssion will require the file to be uploaded. Let us create one.
```python
submission_dict ={"PassengerId":holdout_df["PassengerId"],"Survived":holdout_predictions}
submission = pd.DataFrame(submission_dict)
submission.to_csv("submission_12_15.csv",index=False)
```

Kaggle gives us an accuracy of 79.4% for the holdout data and as of Dec 15th, 2017 the rank for this score is 2179.











