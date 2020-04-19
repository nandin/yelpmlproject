# Introduction
Now more than ever, people do not step foot into a restaurant without copious amounts of research on the internet. Particularly evident in a tourist-friendly city like Las Vegas, restaurant owners continually search for the perfect combination of offerings that will make consumers give them a good review. An investment in better reviews will hopefully lead to a greater influx of consumers. Our study focuses on Yelp, which is rated the most frequented review site. We hope to identify the key factors that contribute to restaurants obtaining a higher score in Las Vegas. By predicting the review score after particular investments, we will help restaurant owners direct their future investments. 

# Visualizing the Data
<p float="left">
  <img src="png_images/nevadaBigMap.png" width="400" />
  <img src="png_images/NevadaSmallMap.png" width="400" /> 
</p>

---
# Cleaning and Pre-processing the Data
## Yelp Dataset -> Las Vegas Restaurant Dataset
The original dataset contained information on 209,393 businesses, not just restaurants, that were found across 11 different metropolitan areas of the world. The data for each business contained the field: **category**. It was determined that to narrow our data to only include information on restaurants, retaining the businesses that contained *Restaurant* or *food* in the *category* field would be acceptable. 

The resulting dataset contained information on 42,152 businesses that presumably all sold food to their customers. In order for the location of these restaurants to not affect the results of any machine learning modeling, it was decided to focus on restaurants contained in only one city. The top 5 cities by count of restaurants is shown below. 

| City | Number of Restaurants|
| :---: | :---:|
|Toronto | 5481 | 
|Las Vegas | 5192| 
|Phoenix | 3133 | 
|Charlotte | 2201 | 
|Montreal | 2013 | 

The decision was made to study a city in the United States so Las Vegas became the location of choice and the dataset was further filtered. 
## Cleaning the Las Vegas Restaurant Dataset
The two main concerns with the restuarant data that needed to be addressed were the completeness of each feature in the dataset and the completeness of each restaurant's data in the dataset. The following procedure was used to clean the data as much as possible. 
1. Features with less than 15% completeness were eliminated from the dataset. Eliminated features are shown below:

    + By Appointment Only
    + Coat Check
    + Drive Thru
    + Smoking
    + Dogs Allowed
    + BYOB
    + Happy Hour
    + Corkage
    + Business Accepts Bitcoin
    + Ages Allowed
    + Accepts Insurance
    + Dietary Restrictions
    + Music
    + Counter Service
    + Best Nights
    + Open 24 Hours
    + Good For Dancing
    + Hair Specializes In

2. Restaurants with less than 80% completeness of data were discarded. 
  
    + Number of Restaurants Kept: 17737
    + Number of Restaurants Eliminated: 24415

3. Features with less than 80% completeness were eliminated from the dataset. Eliminated features are shown below: 

    + Business Accepts Credit Cards
    + Wheelchair Accesible
    + Good for Meal
    + Restaurants Table Service

4. Features that could not be transformed into values that a machine learning model could use as data were deleted next. Eliminated feature are shown below:

    + Name
    + Business Parking
    + Address 
    + Categories 
    + City 
    + Hours 
    + State 
    + Business Id

## Feature Selection
The resulting dataset contained information about 2,503 different restaurants in the Las Vegas area, with each restaurant consisting of data from 22 different features. In order to reduce the number of features in our dataset even further, a correlation matrix was created to determine the relationship between various features and eliminate features with weak correlations to our label. 

<img src="png_images/FeatureCorrelation2Label.png" width="800" />

Some features like *Good for Groups* and *Good for Kids*, which seemed like they would logically have some influence on the rating of a restaurant, ultimately had very little correlation to the number of stars a restaurant recieved. Features, like the *Ambience* and *Price Range* of the restaurant, were suspected of being indicators of the reating of a restaurant. The correlation matrix confirmed that these features had a stronger correlation to the number of stars a restaurant has than other features. An absolute value of a correlation greater than 0.1 was set as the cut off for which features will be kept in the dataset. 

To further reduce the number of features in the dataset, the overall correlation matrix was analyzed in order to determine if any features were strongly related to one or another. If one feature had a strong correlation to another feature, one of them could be discarded and the variance in the data would not decrease significantly. 

<img src="png_images/CorrelationMatrix.png" width = "900" />

The strongest correlation between two features is found to be 0.58 between *Alcohol* and *Price Range*. However, no two features seem correlated enough to each other to reasonably leave them out of the dataset, so no changes are made. 

The final dataset consists of 11 features and 2,503 restaurants. The corresponding label for each restaurant is the star rating they have recieved in yelp. 

---

# Methods
A supervised learning approach for predictive data analysis, specifically tree-based models and regression, were utilized. 

+ Regression: Linear Regression and Ridge Regression
+ Tree-based Models: Decision Tree and Random Forest

The Sci-kit learn packages for the chosen regression and tree-based models were used. A 80 - 20 split was used in order to split the data into a training set and a testing set. 

# PCA/Regression
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

# Decision Tree

# Random Forest Classifier

When running the Random Forest Classifier on the dataset, three parameters were chosen to be explored in determining their effect on the accuracy of the model. The three parameters were: 

1. The number of trees in the forest
2. The criterion for deciding the split when building a tree
3. The maximum depth for each of the trees in the forest

### Number of Trees in the Forest

<img src="png_images/treeVsAccuracy.png" width = "800" />

From the plot above, there seems to be a rise in accuracy from around 21% to around 27% when raising the number of trees in the forest from 1 to 20. After that point, the accuracy oscillates by approximately a percent around 26%, indicating that additional trees do not impact the accuracy of the Random Forest. 

### Split Criterion

<img src="png_images/SplitCriterionEffect.png" width = "800" />

The two splitting criterion that Random Forest Classifier can use are the Gini Impurity and Information Gain Entropy. The number of trees in the model were varied in order to see if any differences in accuracy could be maintained. From the plot above, there seems to be no relationship between the accuracy of the Random Forest model and the splitting criterion it utilizes. 

### Maximum Depth of Trees

<img src="png_images/treeDepthVsAccuracy.png" width = "800" />

A range from 10-100 was examined for the maximum depth of a tree and its effect on the accuracy of the model. The model's accuracy seems to peak at around a depth of 10, before stabilizing around 27 %. Perhaps by minimizing the depth of the tree, the model avoids overfitting the training data and creates better predictions. 

### Results

A Random Forest Classifier was run with the following parameters: 

| Parameter | Value |
| :---: | :---:|
| Number of Trees | 02934|







