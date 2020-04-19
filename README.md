###Repo Link: https://nandin.github.io/yelpmlproject/.

## Introduction
Now more than ever, people do not step foot into a restaurant without copious amounts of research on the internet. Particularly evident in a tourist-friendly city like Las Vegas, restaurant owners continually search for the perfect combination of offerings that will make consumers give them a good review. An investment in better reviews will hopefully lead to a greater influx of consumers. Our study focuses on Yelp, which is rated the most frequented review site. We hope to identify the key factors that contribute to restaurants obtaining a higher score in Las Vegas. By predicting the review score after particular investments, we will help restaurant owners direct their future investments. 

## Visualizing the Data
<p float="left">
  <img src="png_images/nevadaBigMap.png" width="400" />
  <img src="png_images/NevadaSmallMap.png" width="400" /> 
</p>

## Cleaning and Pre-processing the Data

Prithvi to do.

## Methods
We used a supervised learning approach for predictive data analysis, and utilized tree-based models and regression.

Regression: Linear Regression and Ridge Regression
Tree-based Models: Decision Tree and Random Forest

## PCA/Regression
We first used PCA to try and find the optimal variable to begin our regression. However, upon further review, the first principal component was only able to explain BLANK% of the data. The second principal component explained BLANK% of the data. The following graph follows:

Ishita to do: insert graph #k vs explained_variance

We hypothesized after about six principal components, the rmse of the linear regression would flatten, since that is when the explained variance reaches negligible levels. As expected: the graph of the linear rmse against the number of k components follows:

<img src="png_images/linearVpca.png" width="400" />

Running the data through a Ridge Regression follows the same trend:

<img src="png_images/ridgeVpca.png" width="400" />

The rmse's of both regressions were similar and very high, so we can conclude that both regressions are equally bad at predicting the data.
(Ishita to do: Say this better and add specific RMSE's)

However, we can conclude that 6 principal components is that smallest number that still represents the data at an acceptable level.

Ishita to do maybe: Graph breakdown of features that make these up.







