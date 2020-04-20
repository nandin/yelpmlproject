from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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