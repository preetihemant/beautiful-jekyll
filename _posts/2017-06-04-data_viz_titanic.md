---
layout: post
title: Data Visualization in Python - Titanic Disaster
bigimg: /img/CART_tree_titanic_survivors.png
---
## Titanic disaster data is a popular dataset among data enthusiasts. In this post I find answers to questions on passenger survival through python data analysis.

<p> Titanic ship that sailed on 10th April 1912 and sank 5 days later had 2224 passengers aboard. Despite all the safety measures on the ship, 
1502 passengers died leaving only 722 survivors. Here we have a sample data set consisting of the most complete records out of the 
original set. I will be cleaning up this data set, making changes to it and then using it to answer some questions that arise from this
disaster. </p>

<p> We hae some prior knowledge about the accident that can be used to direct our analysis. After the disaster struck, a rescue plan 
was implemented in which women and kids were given first preference on the life boats and hence had better chance at survival. 
We also know that the class of the ticket would have made a difference with a first class passenger been given preference compared
to a third class traveller. Therefore, the important factors in the survival data analysis would be sex, age and pclass. </p>

<p> I will first look at my data set to check for completeness and pre-process it to get it ready for analysis of the questions below. </p>

* [How does the sample set compare to the original data?](#question-1)
* [Men or women, which group had more survivors?](#question-2)
* [How many children survived?](#question-3)
* [Was the strategy of women and children first indeed effective?](#question-4)
* [How many people travelling single survived? Would you advise travelling alone or with family?](#question-5)
* [Does money buy you risk free travel? How many first class passengers survived versus other classes?](#question-6)
* [What age group had the most luck?](#question-7)
* [Based on the above, what kind of a person has the best chance of surviving a similar incident?](#question-8)

## 2. Reading of data

<p> The data set is in the csv format which will be read into a pandas dataframe. We will import some libraries to read the csv file and process/plot data </p>

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename_train = (file_location)
train = pd.read_csv(filename_train)
```
## 3. Data Wrangling

### 3.1 Handling of missing data

<p> The training set has 891 records with 12 columns or features such as Pclass, Sex, Age and so on. There are some missing values in the Age, Cabin and station Embarked category </p>

```python
print train.shape  #  DataFrame.shape to calculate the number of rows and columns
print train.info()

(891, 12)
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
```
* Cabin number would be important in analysis of the effect of social class but since we have complete data for class of the ticket, we can ignore the missing cabin numbers. 
* Missing station data from where the passengers embarked can also be ignored since the disaster took place after all the passengers were aboard.
* Missing age data however needs to be handled correctly. We will assign a value of -0.5 to the missing age data and categorize it as "missing" later on
```python
train["Age"] = train["Age"].fillna(-0.5)
```
### 3.2 Data mapping

<p> An age below 1 is entered as a fraction. Age is a continuous series. Since sex and pclass are categorical variables, age can also be split into categories. We will pick age groups as infant, child, teenager, young adult, adult , senior and missing and store this in a new column "Age_categories" </p>

```python
print train["Age"].describe()
count    714.000000
mean      29.699118
std       14.526497
min        0.420000
25%       20.125000
50%       28.000000
75%       38.000000
max       80.000000
Name: Age, dtype: float64

cut_points=[-1,0,5,12,18,35,60,100]
labels=["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
train["Age_categories"] = pd.cut(train["Age"],cut_points,labels)
```

<p> I will change the ticket class to First, second and third corresponding to 1, 2 and 3 in the data set. </p>

```python
map_pclass={1:'First class',2:'Second class',3:'Third class'}
train["Pclass"] = train["Pclass"].map(map_pclass)
```

<p> I will also change the survived data to boolean </p>

```python
map_survive = {0:False,1:True}
train["Survived"] = train["Survived"].map(map_survive)
```

<p> I will add a new column "company" that tells us if a traveller was alone or had family (parents, sibling, spouse, children).To compute this I will change the SibSp and Parch to boolean. </p>

```python
train["SibSp"]=train["SibSp"].astype('bool')
train["Parch"]=train["Parch"].astype('bool')
train["company"] = train["SibSp"] | train["Parch"]
```

## 4. Analysis

* ### How does the sample set compare to the original data?

<p> The original data set has a record of 2224 passengers out of which 1502 died and 722 survived. Our sample set has almost the same survival/death data as the original and can be considered a sufficiently accurate sample for our analysis. </p>

```python
from tabulate import tabulate

org_passengers = 2224
org_survivors = 722
org_died = 1502

sample_passengers = len(train)
sample_survivors = len(train[train.Survived == True])
sample_died = sample_passengers - sample_survivors

table = [["Total",org_passengers," ",sample_passengers," "],["Survivors",org_survivors,org_survivors/float(org_passengers)*100,sample_survivors,sample_survivors/float(sample_passengers)*100],["Died",org_died,org_died/float(org_passengers)*100,sample_died,sample_died/float(sample_passengers)*100]]
print tabulate(table, headers = ["  ","Original","Original %","Sample","Sample %"],tablefmt="fancy_grid")
```
  
   |         | Original| Original %   | Sample  | Sample %      |
   | :-----: | :-----: |:-----:       | :-----: | :-----:       |
   | Total   |  2224   | 891          |         |               |
   | Survivors | 722   | 32.464028777 | 342     |  38.3838383838|
   | Died    | 1502    | 67.535971223 | 549     | 61.6161616162 | 
   
&nbsp;
* ### Men or women, which group had more survivors?

<p> Women were given first preference on the life boats along with children and hence more women were expected to survive. Our data too shows the same result. 74.2% of the female passengers survived while only 18.9% of the male passengers managed to remain alive. This is consistent with the women/children first rescue strategy. </p>

```python
gender = pd.crosstab(train["Sex"],train["Survived"])
print gender
gender.plot.bar(stacked = True)

gender_percentage=train.pivot_table(index="Sex",values="Survived")
print gender_percentage
gender_percentage.plot.bar()
plt.show()
```

|Survived by numbers| Died   | Survived |Survived by percentage|
|:-----------:      |:--:|:-:|     :----------------:          |
|Sex                |    |   |               |  
|female             |81  |233|   0.742038             |
|male               |468 |109|   0.188908             |


![alt text](https://raw.githubusercontent.com/preetihemant/Data_Science_Projects/master/Titanic_analysis/Male_vs_Female.png "Male vs Female survival")
![alt text](https://raw.githubusercontent.com/preetihemant/Data_Science_Projects/master/Titanic_analysis/male_vs_female_percentage.png "Male vs Female survival percentage")
&nbsp;
* ### How many children survived? 
<p> Close to 54% of the children (less than 18 years old) survived </p>

```python
children = pd.crosstab(train["Survived"],train["Age"]<18)
print children
children[True].plot.bar()
plt.show()
```

![alt text](https://raw.githubusercontent.com/preetihemant/Data_Science_Projects/master/Titanic_analysis/children_survived.png "Survival data for children")

```python
num_children = (train["Age"]<18).sum()
num_survived = gender[True][True]
child_percentage= num_survived/float(num_children) * 100
print child_percentage
```

| Age | False | True|
| :--: | :---:| :---:|
|Survived |  |   |
|0 | 497 | 52|
|1 | 281 | 61 |

&nbsp;
* ### Was the strategy of women and children first indeed effective?

<p> Considering the women and children survival rates individually, it is hard to say if the resuce operations were effective, since this ship was supposed to be sink-proof with sufficient life boats/jackets and a good emergency plan. However considering how low the survival numbers were for the male gender, higher proportion of women and children surviving can only be attributed to the fact that they were given first preference. So, based on the data we have, the strategy was effective. </p>
&nbsp;
* ### How many people travelling single survived? Would you advise travelling alone or with family?

<p> One would think travelling alone is a better option in a scenario like this since one has to only lookout for herself or himself. However, our data tells a different story.
The probability of a traveller with family surviving is about the same as dying, while for a single traveller the probability of dying(69.6%) greatly increases. </p>

```python
alone = pd.crosstab(train["company"],train["Survived"])
print alone
alone.plot.bar()
plt.show()
```

| Survived |0 | 1|
| :--: | :---:| :---:|
|Company |  |   |
|False | 374 | 163|
|True  | 175 | 179 |


![alt text](https://raw.githubusercontent.com/preetihemant/Data_Science_Projects/master/Titanic_analysis/alone_survived.png "Survival date for people travelling alone")
&nbsp;
* ### Does money buy you risk free travel? How many first class passengers survived versus other classes?
 
 <p> In the third class ticket category, we can see that a lot many died than survived. Among first class travellers, more survived than died, whereas second class passengers had a roughly equal chance at survival. Hence, being able to afford a first class ticket most likely also shields you from misfortune. This is consistent with what we see in our society with the rich and mighty. Money can buy you safety among other things. </p>
 
 ```python
pclass = pd.crosstab(train["Pclass"],train["Survived"])
print pclass
class_categories=["First class","Second class","Third class"]
pclass.plot.bar()
plt.xticks([0, 1, 2],class_categories)
plt.show()
```

| Survived |0 | 1|
| :--: | :---:| :---:|
|Pclass |  |   |
|1 | 80 | 136|
|2  | 97 | 87 |
| 3| 372 | 119 |

![alt text](https://raw.githubusercontent.com/preetihemant/Data_Science_Projects/master/Titanic_analysis/pclass_survived.png "Impact of ticket class on survival")
&nbsp;
* ### What age group had the most luck?

<p> From the analysis, women seemed to have better survival rates across all age groups. In particular, women in the young adult age group had the most luck. </p>

```python
age_group_gender = pd.crosstab(index=train["Age_categories"], columns=[train["Sex"],train["Survived"]])

age_group_gender["female"].plot.bar()
plt.xticks([0, 1, 2,3,4,5,6],age_categories)
plt.subplot().set_title("Proportion of survivors by age group / Female",fontsize=14)

age_group_gender["male"].plot.bar()
plt.xticks([0, 1, 2,3,4,5,6],age_categories)
plt.subplot().set_title("Proportion of survivors by age group / Male",fontsize=14)
plt.show()
```
![alt text](https://raw.githubusercontent.com/preetihemant/Data_Science_Projects/master/Titanic_analysis/female_age_category_survival.png "Female survival data by age group")

![alt text](https://raw.githubusercontent.com/preetihemant/Data_Science_Projects/master/Titanic_analysis/male_age_group_survival.png "Male survival data by age group")
&nbsp;
* ### Based on the above, what kind of a person has the best chance of surviving a similar incident? 

<p> From all the questions answered so far, a rich/upper class, yound adult female accompanied by somebody is the best category to be in to survive a similar incident </p>

## 5. Conclusion

### 5.1 Incompleteness of data

  * The data is only a subset of the full passenger information. Out of the 2224 travellers, we have fairly complete information only for about 819. This limits our ability to draw accurate conclusions.
  * There are missing values for age which happens to be an important feature. I have treated the missing data as missing itself. However, one could use other methods and average out the missing ages. This would produce slightly different results from what I have here. However, one method isn't more correct than the other. Depending on our application and the amount of tolerance we wish, one of the several methods for missing data can be chosen.
 * The given data does not distinguish between passengers and crew members. Hence we do not know how many crew members survived. However, knowing that most crew members at the time would have been male and not many male passengers survived, it would be fair to say not many crew members survived.
