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
<p> </p>

#### Best performing features
At the end of all the steps of feature prepartion, we now have over 30 features. Surely some of them must be correlated, some other irrelevant. We could always feed all of them into our ML model, but that will increase training time and complexity. Let us instead cut this list down to the best performing features.







