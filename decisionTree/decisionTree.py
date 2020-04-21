from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
# Decision Tree

In this method, we wanted to see if a regressor being could yield a better result based on the continous nature of the Stars ratings. With that, we had to drop PostalCode as a feature as it wouldn't fit the nature of the prediction model. 

For decision trees, we used the scikit implementation of the regression model and attempted two variations:

1. Maximum Depth allowed in the tree
2. Boosting the Tree

 ## Maximum Depth of Trees and Stablizing

<img src="rmsecomparor.png" width="800" />

Maximum Depth: At first, we experimented with a depth level of 10 and observed how the calculate errors began to go down till it hit a minimum of 0.767078 at the max depth of 5 trees. We decided to use this max depth for both DecisionTreeRegressor and AdaboostRegressor to maintain consistency and because the difference between the AdaboosRegressor min and what was shown at max depth of 5 trees was minimal.

We used RMSE for calculating error. Used r2_score for calculating variance.

We used the Adaptive Boosting (AdaBoost) regressor (scikit implementation) that is essentially increasing the weight of misclassified data points and updating,  then making a new prediction by by adding the weights of each tree times the  prediction of each tree. We hypothesized that boosting would lower our rmse. We used the maximum depth that was used in the previous model to see if boosting had actually improved the model.

<img src="decisiontreeregressors.png" width="800" />

As seen, the boosting did help reduce error, although not significantly. Other methods like pruning were attempted but the tree proved too sensitive to run many of the pruning methods. 

Despite the Adaboost regressor yielding better results, we feared that the decision tree regression had overfit the data so we decided to see if a classification of the star ratings in a random forest classifier would yield better results.
"""
def main():
	# load the dataset
	data = np.load('FinalDataSet.npy')
	data = data[~pd.isnull(data).any(axis=1)]
	data = data.astype('int')
	data = data[:,1:] #remove business ID
	stars = data.T[11]
	data = (data.T[0:11]).T

	trainData, testData, trainStars, testStars = split(data, stars)

	easierTrainStars = np.where(trainStars > 5, 1, 0)
	easierTestStars = np.where(testStars > 5, 1, 0)

	# see effect of number of trees in random forest on data
	accuracy = np.zeros(100)
	accuracyEasy = np.zeros(100)


	# Optimizing the number of trees in the forest
	for i in range(1):
		model = DecisionTreeClassifier()
		
		modelEasy = DecisionTreeClassifier()

		model.fit(trainData, trainStars)
		modelEasy.fit(trainData, easierTrainStars)
		
		accuracy[i] = model.score(testData, testStars) * 100
		print(accuracy[i])
		accuracyEasy[i] = modelEasy.score(testData, easierTestStars) * 100
		print(accuracyEasy[i])

	legend = ('Multi-Class Label', 'Binary Label')
	# two_plot(accuracy, accuracyEasy, "Number of Trees", 'Accuracy (%)', 'Accuracy Difference Between Binary and Multi-Class Labels', legend)

def two_plot(accuracyOne, accuracyTwo, xAxis, yAxis, title, legend):
	xAxisArr = np.arange(1, 101, 1)
	plt.style.use('seaborn')
	classOne = plt.scatter(xAxisArr, accuracyOne, c = (0.27, 0.58, 0.37), s  = 30)
	classTwo = plt.scatter(xAxisArr, accuracyTwo, c = (.15, 0.35, 0.58), s = 30)
	plt.xlabel(xAxis)
	plt.ylabel(yAxis)
	plt.title(title)
	plt.legend((classOne, classTwo), legend)
	plt.tight_layout()
	plt.show()


def split(arr, labels):
	trainingPercentage = 0.7
	n, d = arr.shape
	trainX = arr[0:int(trainingPercentage * n), :]
	testX = arr[int(trainingPercentage * n):]
	trainY = labels[0:int(trainingPercentage * n)]
	testY = labels[int(trainingPercentage * n):]

	return trainX, testX, trainY, testY

if __name__ == "__main__":
    main()