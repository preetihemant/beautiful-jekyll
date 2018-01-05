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
![histogram](img/species_hist.png)

Box plots give us the spread of the data and the species histogram tells us the smaller petal widths are indeed real observations, they all belong to one particular species - Iris setosa 
