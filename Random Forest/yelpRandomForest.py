from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
data = np.load('dataset.npy')
data = data.astype('int')
numDataPoints = np.size(data, 0)

# Split data set into training and testing -> 80% training 20% testing
eightyPerc = int(np.around(numDataPoints * 0.8))
trainingData = data[:eightyPerc, 1:10]
testingData = data[eightyPerc:, 1:10]

# Split labels into training and testing
trainingLabels = np.around(data[:eightyPerc, 10])
testingLabels = np.around(data[eightyPerc:, 10])

# see effect of number of trees in random forest on data
accuracy = np.zeros(1000)
for i in range(1000):
	model = RandomForestClassifier(n_estimators = i+1, 
									bootstrap = True, 
									criterion = 'entropy', 
									max_features = 'sqrt')
	model.fit(trainingData, trainingLabels)
	accuracy[i] = model.score(testingData, testingLabels)

numberTrees = np.arange(1, 1001, 1)
plt.scatter(numberTrees, accuracy)
plt.show()


