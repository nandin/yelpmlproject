import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  

def main():
	f = open('FinalDataSet.npy', 'rb') 

	data = np.load(f)
	data = data[~pd.isnull(data).any(axis=1)]
	stars = data.T[12]
	data = (data.T[2:12]).T

	# pca over all the components
	linearAccuracy = np.zeros(10)
	ridgeAccuracy = np.zeros(10)
	for i in range(1, 11):
		pca = PCA(n_components = i)
		pca.fit(data)
		components = pca.transform(data)

		# split data -> 80% training 20% testing
		trainData, testData, trainStars, testStars = split(components, stars)

		# Linear Regression w first i PCA components
		regressor = LinearRegression()  
		regressor.fit(trainData, trainStars) #training the algorithm

		predStars = regressor.predict(testData)
		rmseResult = rmse(predStars, testStars)
		linearAccuracy[i - 1] = rmseResult

		# Ridge Regression w first i PCA components
		clf = Ridge()
		clf.fit(trainData, trainStars)

		predStars = clf.predict(testData)
		rmseResult = rmse(predStars, testStars)
		ridgeAccuracy[i - 1] = rmseResult

	# visualize
	numberComponents = np.arange(1, 11, 1)
	plt.xlabel("Number of Components")
	plt.scatter(numberComponents, linearAccuracy)
	plt.ylabel("Linear RMSE")
	plt.show()
	plt.scatter(numberComponents, ridgeAccuracy)
	plt.xlabel("Number of Components")
	plt.ylabel("Ridge RMSE")
	plt.show()


def split(arr, labels):
	trainingPercentage = 0.7
	n, d = arr.shape
	trainX = arr[0:int(trainingPercentage * n), :]
	testX = arr[int(trainingPercentage * n):]
	trainY = labels[0:int(trainingPercentage * n)]
	testY = labels[int(trainingPercentage * n):]

	return trainX, testX, trainY, testY


def rmse(pred, label): 
    N = pred.shape[0]

    difference = ((pred - label) ** 2) / N
    rmse = np.sqrt(difference.sum())

    return rmse


if __name__ == "__main__":
    main()