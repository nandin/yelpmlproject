import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  

def main():
	data = np.load('FinalDataSet.npy')
	data = data[~pd.isnull(data).any(axis=1)]
	data = data.astype('int')
	data = data[:,1:] #remove business ID
	stars = data.T[11]
	data = (data.T[0:11]).T
	# plotVariance(data)
	# lasso(data, stars)

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

	plotRegressions(linearAccuracy, ridgeAccuracy)
	# plotVariance(data)

def lasso(data, stars):
	trainData, testData, trainStars, testStars = split(data, stars)
	clf = Lasso()
	clf.fit(trainData, trainStars)
	predStars = clf.predict(testData)
	rmseResult = rmse(predStars, testStars)
	print(rmseResult)

def split(arr, labels):
	trainingPercentage = 0.7
	n, d = arr.shape
	trainX = arr[0:int(trainingPercentage * n), :]
	testX = arr[int(trainingPercentage * n):]
	trainY = labels[0:int(trainingPercentage * n)]
	testY = labels[int(trainingPercentage * n):]

	return trainX, testX, trainY, testY

def plotRegressions(linearAccuracy, ridgeAccuracy):
	numberComponents = np.arange(1, 11, 1)
	plt.style.use('seaborn')
	Linear = plt.scatter(numberComponents, linearAccuracy, c = (.15, 0.35, 0.58))
	plt.xlabel('Number of Components')
	plt.ylabel('RMSE')
	plt.title('Number of Components vs. RMSE (Linear)')
	plt.tight_layout()
	plt.show()

	Ridge = plt.scatter(numberComponents, ridgeAccuracy, c = (0.27, 0.58, 0.37))
	plt.xlabel('Number of Components')
	plt.ylabel('RMSE')
	plt.title('Number of Components vs. RMSE (Ridge)')
	plt.tight_layout()
	plt.show()

def plotVariance(data): 
	pca = PCA()
	pca.fit(data)
	variances = pca.explained_variance_ratio_
	numberComponents = np.arange(1, 11, 1)
	plt.style.use('seaborn')
	plt.scatter(numberComponents, variances, c = (0.15, 0.35, 0.58), s = 30)
	plt.xlabel('Number of Principal Components')
	plt.ylabel('Explained Variance Ratio')
	plt.title('Variances Explained by Principal Components')
	plt.tight_layout()
	plt.show()

def rmse(pred, label): 
    N = pred.shape[0]

    difference = ((pred - label) ** 2) / N
    rmse = np.sqrt(difference.sum())

    return rmse


if __name__ == "__main__":
    main()