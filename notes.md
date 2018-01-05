---
layout: page
title: Concepts explored
---

  * [68-95-99.7 Rule](#68-95-99.7_Rule)
  * [Mean, mode, median - a comparison](#3_M's_quick_reference)   
  * [Correlation and Causation - Golden Arches theory of conflict prevention](#Correlation and Causation)
  * [Observational Studies/Surveys](#Observational studies)
  * [Controlled Experiments](#Controlled experiments)
  * [Klout Social media analytics](#Klout analytics)
  
### 68-95-99.7_Rule
The rule gives a quick estimation of the distribution of data. Given a distribution with mean μ and std. deviation σ, 68% of the values lie within one standard deviation of the mean. 95% of the data within 2 standard deviations of the mean and 99.7% between 3 standard deviations of the mean.

* P(μ  - σ <= X <= μ  + σ) ~ 0.68
* P(μ - 2σ <= X <= μ + 2σ) ~ 0.95
* P(μ - 3σ <= X <= μ + 3σ) ~ 0.99
                   
![alt text](https://raw.githubusercontent.com/preetihemant/Udacity/master/Descriptive_Statistics_Course/68-95-99.7_rule.png "Normal Distribution Proportions")

### 3_M's_quick_reference
How do the three measures of central tendency compare? Here is a list of factors and how they affect the mean, mode and median

![alt text](https://raw.githubusercontent.com/preetihemant/Udacity/master/Descriptive_Statistics_Course/3M's.png "Central_measure_comparisons")

* Mean has a simple equation Mean = (Sum of all data values)/(No. of Data values)
* Median too can be described by an equation but it is not as simple as the average
  * For even n [ X <sub>n/2</sub> + X <sub>(n/2)+1</sub> ] / 2
  * For odd  n  X <sub>(n+1)/2</sub>
* Although median and mode are dependent on the data values, they are not severely affected by small changes. For instance both mode and     mean are unaffected by the addition of a single new value.
* If looking at a histogram to determine mode, then mode will take on different values depending on the bin size. The maximum value of mode   occuring when there is just one bin.
- I rock a great mustache
- I'm extremely loyal to my family

### Correlation and Causation
<p>Two variables are said the be correlated if there appears to be a relationship between them. As one of them changes, the other shows a change too. For instance, the amount of water collected in a fresh water lake varies as the amount of rainfall received. Assuming there is no other source of water filling up the lake, there is a direct correlation between the amount of water in the lake at a given time and the rainfall received. Rainfall is the cause and water level is the effect. These two variables are said to be a causation pair </p>
<p>Now, consider another example - plant growth in the same area and rainfall received. Although we expect a higher rate of plant growth with increase in rainfall, there are many other factors affecting flora and fauna. Plants also require good soil, manure and sunlight to grow. Hence, we cannot establish the cause-effect relationship in our second example. Rainfall and plant growth are correlated but not causal. </p>

#### Golden Arches Theory of Conflict Prevention
<p> Thomas L. Friedman in his book "The Lexus and the Olive Tree" descibes a capitalist theory known as the Golden Arches Theory of Conflict Prevention. </p>
<p> "No two countries that both had McDonald's had fought a war against each other since each got its McDonald's." </p>
<p>The theory through observation claims that any two countries with McDonald's in them would never go to war, in other words a country's war policy is correlated to the presence of McDonald's chain. If this is true, can we say that establishng McDonald's in every country would ensure world peace? Not true! The theory is only an observation and has tied two factors that may be or may not be causal. 
One explanation for the observed correlation is that a country opening its doors to McDonald's also very likely has open economy and hence would likely avoid war. But it would be wrong to say that having this store chain changes a nation's foreign policy. </p>

## Observational studies 
<p> Observational studies are used to establish relationships between two constructs or variables. The environment under study is not controlled there by making the data obtained independent of how the data came about. </p>
<p> The independent variable is called the predictor variable while the dependent variable is the outcome. As a first step, an operational definition is established to measure the two variables. As an example, one of the many operational definitions of sleep quality can be the number of hours of sleep. Next, the changes in the outcome variable are observed and recorded as a respone to changes in the predictor variable. The relationship between the two can be visualized as a scatter plot. </p>

#### Candidates for observational studies:

*  Study of the relationship between season(predictor) and sale of hiking gear(outcome)
*  Variation in the annual rainfall(outcome) and green cover(predictor)
*  Changes in employee benefits(outcome) versus employer taxation (predictor)
*  Internet usage(outcome) versus time of the day(predictor)
*  Relation between crime rate(outcome) and unemployment (predictor)

## Surveys
<p> Surveys provide an easy, inexpensive way to collect data for questions we are looking to answer. They also form an important tool to establish correltion between constructs.For instance, a survey could be setup to gather data on spending habits or social media interaction of a group of people. </p>
<p> Although surveys can be remotely setup and can be made openly accessible, they come with downsides. One of the issues is untruthuful or biased response from participants. It is also possible that the respondents do not understand the questions, known as response bias. On the other hand, respondents may refuse to answer, known as non response bias. Surveys should be built in a way to mitigate the effects of these errors. </p>

## Correlation does not imply causation

<p> An important aspect of Observational studies or surveys is that they can only be used to show relationship. They cannot and should not be a means to establish causation. </p>
<p> This is due to the presence of extraneous factors virtually in every situation. Also know as lurking variables, they have a significant impact on the outcome variable. Unless these factors are neutralized, establishing direct relationship between the predictor and outcome would produce misleading results. However, since the lurking variables are constructs, measuring them is tricky. Also, it is very hard to factor in all the extraneous influences. These limitations make observational studies only useful in predicting a trend. <p>

## Controlled experiments

<p> While observational studies and surveys help us establish correlation, controlled experiments let us determine causal relationships.
In a typical controlled expermient, one variable is changed at a time keeping all others constant. Care is also taken to include extraneous variables into account. Since there is only one factor varying at a time, a direct relationship can be established between the changing variable (predictor) and the result (outcome). This helps us understand the impact of each predictor variable on the outcome and may be used to discard certain factors that have little effect on the result. 

## Klout analytics
<p> Klout is a mobile application and a webpage that assigns ratings or ranking to its users based on their social media presence. It uses different social media platforms like facebook, twitter, instagram to determine user activity and social influence. A higher rating would mean greater online social media impact. Airlines, hotels could use somebody's klout score to decide on upgrades and services much like credit card companies using one's credit score. </p>

<p>Klout analytics tells us how data analytics and algorithms are applied in real world to quantify constructs. It also brings up the very important question of correlation implying causation. A high klout score is certainly indicative of ones online impact but is it the only measure? Is it also a completely reliable and accurate measure? </p>

<p>Klout scores are calculated using algorithms that use data points such as number of followers, likes, retweets from twitter, facebook and some other apps. Applications like Flickr, Wordpress are not counted towards the score. A user with a considerable Flickr and worpress influence may have a high social impact but would not be assigned a high rating making the klout score some what misleading. </p>

<p>In case of twitter, a high impact profile will typically have thousands of followers, this user however may only be following a handful of other accounts. How these factors are weighed in the algorithm is very important.</p>

<p>There also are some tricks floating around on the web that can increase your klout score. If these indeed work, then what good is a scoring system that can be gamed? </p>

<p>Analytics such as klout do a great job of data crunching and bringing meaning out of a large dataset but a closer look tells us that the conclusions drawn can be flawed, biased and misleading at times. </p>


