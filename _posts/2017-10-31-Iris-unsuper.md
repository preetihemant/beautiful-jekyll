---
layout: post
title: UnSupervised Machine Learning
subtitle: Iris Dataset
image: /img/Iris_versicolor_3.jpg
bigimg: /img/iris.jpg
---

<p> In unsupervised learning, samples are classified without the knowledge of their labels. K means clustering is a popular 
classification algorithm for unsupervised learning. Here, we group data into clusters of similar samples. Each cluster has a
centroid,the algorithm finds the nearest centroid for each data point and assign it to that cluster. The number of centroids or 
classes we want should be specified before the algorithm is run. We can employ the elbow method to find the optimal number of classes.
In case of the Iris data set we already know the number of claases which is 3 - Iris setosa, versicolor and virginica</p>

An exploratory analysis of the iris data can be found in my earlier post [Supervised Machine Learning - Iris data](https://raw.githubusercontent.com/preetihemant/preetihemant.github.io/master/_posts/2017-10-29-iris-supervised.md)

<p> Let us first train our Kmeans model using all the data we have and not split it into test-train. We will use the same two features we used in supervised learning, the only change here is that we train our model without using any classification labels. </p>

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3,random_state = 0)
kmeans.fit(features)
predictions_kmeans = kmeans.predict(features)
```

<p> Now that we have trained our model and used it predict classes, let us visualize the classification and compare it with the original classified data.</p>

```python
import numpy as np
import matplotlib.patches as mpatches
plt.figure(figsize=(14,7))
colormap=np.array(['blue','green','red'])
blue = mpatches.Patch(color='blue',label='iris setosa')
green = mpatches.Patch(color='green',label='iris versicolor')
red = mpatches.Patch(color='red',label='iris virginica')
black = mpatches.Patch(color = 'black',label ='Centroids')

plt.subplot(1, 2, 1)
plt.scatter(iris_df["PetalWidthCm"], iris_df["SepalWidthCm"], c= colormap[predictions_kmeans], s=40)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('Petal Width')
plt.ylabel('Sepal Width')
plt.title('Predicted Classification')
plt.legend(handles=[blue,green,red,black])

plt.subplot(1, 2, 2)
plt.scatter(iris_df["PetalWidthCm"], iris_df["SepalWidthCm"], c= colormap[iris_df["Group"]], s=40)
plt.xlabel('Petal Width')
plt.ylabel('Sepal Width')
plt.title('Real Classification')
plt.legend(handles=[blue,green,red])
plt.show()
```

![Predicted classes 1](/img/unsup_class_1.png)
<p> We can see that our model does a good job of classifying samples except that group 1 and 2 have the classification label interchanged. Let us change the predicted labels from 1's to 2's and 2's to 1's to give the correct number to each group. Now when we plot the points again, we see the right labels as the original data. Except for a few wrongly classified samples, the model has done a pretty neat job. </p>

```python
predictions_kmeans = np.choose(kmeans.labels_, [0, 2, 1]).astype(np.int64)
```

![Predicted classes 2](/img/unsup_class_2.png)

<p> The classification looks good when visualized, but is it really? We will need to evaluate performance mathematically. To calcualte our model's peroformance on unknown data, let us split our input into test and train. We can then use the test data to calculate the accuracy. Also, since this is an unsupervised learning exercise, we will get better results as the training data increases. Hence this time I will do that 80-20 split of the samples. </p>

```python
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=14)

kmeans_train = KMeans(n_clusters = 3,random_state = 14)
kmeans_train.fit(features_train,labels_train)
pred_kmeans_train = kmeans_train.predict(features_test)

### Performance
accuracy_kmeans = accuracy_score(pred_kmeans_train,labels_test)
precision = precision_score(labels_test, pred_kmeans_train, average="weighted")
recall = recall_score(labels_test, pred_kmeans_train, average="weighted")
print ("Accuracy kmeans:",accuracy_kmeans)
print ("Precision kmeans:", precision) 
print ("Recall kmeans:", recall) 
```

<p>We get an accuracy of 93.34%, although it is less than the accuracy we obtained using Logisitc regression and knn models and that we used 80% of our data to train versus 70% in the other two models, the performance is still very good for unsupervised learning.
Knowing the number of clusters in our data made it easier to achieve good performance. In most practical problems, the number of clusters is an unknown an turns out to be the most challenging aspect to tune. </p>










