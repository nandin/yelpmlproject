# Introduction
Now more than ever, people do not step foot into a restaurant without copious amounts of research on the internet. Particularly evident in a tourist-friendly city like Las Vegas, restaurant owners continually search for the perfect combination of offerings that will make consumers give them a good review. An investment in better reviews will hopefully lead to a greater influx of consumers. Our study focuses on Yelp, which is rated the most frequented review site. We hope to identify the key factors that contribute to restaurants obtaining a higher score in Las Vegas. By predicting the review score after particular investments, we will help restaurant owners direct their future investments. 

# Visualizing the Data
<p float="left">
  <img src="png_images/nevadaBigMap.png" width="400" />
  <img src="png_images/NevadaSmallMap.png" width="400" /> 
</p>

---
# Cleaning and Pre-processing the Data
The original dataset contained information on 209,393 businesses, not just restaurants, that were found across 11 different metropolitan areas of the world. The data for each business contained the field: *category*. It was determined that to narrow our data to only include information on restaurants, retaining the businesses that contained *Restaurant* or *food* in the *category* field would be acceptable. 

The resulting dataset contained information on 42152 businesses that presumably all sold food to their customers. In order for the location of these restaurants to not affect the results of any machine learning modeling, it was decided to focus on restaurants contained in only one city. The top 5 cities by count of restaurants is shown below. 

| City | Number of Restaurants|
| :---: | :---:|
|Toronto | 5481 | 
|Las Vegas | 5192| 
|Phoenix | 3133 | 
|Charlotte | 2201 | 
|Montreal | 2013 | 

---

# Methods
We used a supervised learning approach for predictive data analysis, and utilized tree-based models and regression.

Regression: Linear Regression and Ridge Regression
Tree-based Models: Decision Tree and Random Forest

