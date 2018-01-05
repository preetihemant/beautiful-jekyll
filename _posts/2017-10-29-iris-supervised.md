---
layout: post
title: Supervised Machine Learning
subtitle: Iris Dataset
image: /img/Iris_versicolor_3.jpg
bigimg: /img/iris.jpg
---

The Iris flower data set was first used by Ronald Fisher in his paper in 1936. It is popular in the machine learning and data science community today. This classic datset is simple, well understood and will give a good ML introduction to anybody just starting. 

## About the data set
150 data samples split into 3 species of Iris -  Iris setosa, Iris virginica and Iris versicolor. Each sample has four attirbutes - Sepal width, Sepal length, Petal width, Petal length. All the samples have classification labels and this would make this data a supervised learning exercise.

## Data exploration
As a first task, I will explore the data I am working on - data dimensions and data types, missing data points, anomalies.
As a next step, I will visualize the data to bring out any more insights and use these to determine features for the ML algorithm.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)

iris_df = pd.read_csv("Iris.csv")

print iris_df.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 6 columns):
Id               150 non-null int64
SepalLengthCm    150 non-null float64
SepalWidthCm     150 non-null float64
PetalLengthCm    150 non-null float64
PetalWidthCm     150 non-null float64
Species          150 non-null object
dtypes: float64(4), int64(1), object(1)
memory usage: 7.1+ KB
None
```
```python
print iris_df.describe()

               Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
count  150.000000     150.000000    150.000000     150.000000    150.000000
mean    75.500000       5.843333      3.054000       3.758667      1.198667
std     43.445368       0.828066      0.433594       1.764420      0.763161
min      1.000000       4.300000      2.000000       1.000000      0.100000
25%     38.250000       5.100000      2.800000       1.600000      0.300000
50%     75.500000       5.800000      3.000000       4.350000      1.300000
75%    112.750000       6.400000      3.300000       5.100000      1.800000
max    150.000000       7.900000      4.400000       6.900000      2.500000
```

```python
print iris_df["Species"].value_counts()

Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: Species, dtype: int64
```

I see that there are 50 samples for each species and there are no missing values in the features. The values look correct except for petalwidth whose minimum value of 0.1 stands out compared to other minimums. We will need to examine this further to make sure the low minimums are real data points.

```python
data = [iris_df["PetalLengthCm"],iris_df["PetalWidthCm"],iris_df["SepalLengthCm"],iris_df["SepalWidthCm"]]
plt.boxplot(data)
label_names = ("PetalLengthCm   PetalWidthCm   SepalLengthCm   SepalWidthCm")
plt.xlabel(label_names)
sns.FacetGrid(iris_df, hue="Species", size=5).map(plt.hist,"PetalWidthCm",bins=10).add_legend()
plt.show()
```
![box plot](/img/box_plot_features.png)
![histogram](/img/species_hist.png)

Box plots give us the spread of the data and the species histogram tells us the smaller petal widths are indeed real observations, they all belong to one particular species - Iris setosa 

## Visualization of the features

```python
# Scatter plots
sns.FacetGrid(iris_df, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
sns.FacetGrid(iris_df, hue="Species", size=5).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()
# Pair plots
sns.pairplot(iris_df.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
plt.show()
```
![scatter sepal plot](/img/sepal_scatter_plot.png)
![scatter petal plot](/img/petal_scatter_plot.png)
![pair plot](/img/pair_plot.png)

From the plots, the species iris-setosa is linearly separble from the other two species. There is some overlap in iris - versicolor and virginica. This might cause some errors and a decrease in our machine learning model. Since the 3 groups of data are separable to a large degree, we can apply classification models with 3 classes.

## Machine Learning Process
### 1. Feature preparation

In our data, Species is a categorical vairable, let us convert this to a numerical value to make it easier to use in ML and data visualization.
```python
def convert_species(species):
    if species == "Iris-setosa":
        return 0
    elif species == "Iris-versicolor":
        return 1
    else:
        return 2
       
iris_df['Group']=iris_df['Species'].apply(convert_species)
```
The newly created Group column also serves as the classification label data for our sample points. Lets store the labels to feed into our model
```python
labels = iris_df["Group"]
```
### 2. Feature selection

<p> In the petal width/length plot, there is a species wise grouping of data points. Although the classification boundary between versicolor and virginica is not distinct, it is roughly linear. 
Sepal length/width also groups the data but there is a significant overlap between the versicolor and virginica species. </p>
<p> We could use the pair plots to determine relationship between every combination of features. From these plots, petal lenght/width, (petal length - sepal length) and (petal width - sepal width) group the species well without a significant overlap. </p>
<p> We are limiting the number of features to 2, keeping this in mind our best two features would be petal width and sepal width. The reason I selected petal width over petal length is that it includes the low range of values around 0.1cm. </p>
<p> There are mathematical ways of finding the features with most information using RFE - Recursive Feature Elimination.  Since this is a relatively simple data set, I will not complicate the ML process by introducing RFE. If my algoithm accuracy is not as high as desired, I will reconsider RFE. </p>

## 3. Train/test data split
We will split our data set into train and test set. The test data is unseen and will be used to evaluate the performance of our algorithm. If we train and test on the same data, accuracy will always be high and we will not know how our algorithm performs with new and unknown data. To split our data, we will use the test train split method in sklearn

```python
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=14)
```
It is important to assign some number to random_state variable to ensure our results are reproducible.

## 4. Model selection
Let us first start with a popular binary classification algorithm Logistic regression but can be extended to more than two classes. Using the petal width and sepal width as our features, we can predict the classification of unknown iris petal/sepal values.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression()   
lr.fit(features_train,labels_train)
```
