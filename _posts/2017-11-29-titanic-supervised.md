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
visualizing data for any anomalies,correlations.
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

<p> Let us now apply this process to the titanic survival data set. </p>


