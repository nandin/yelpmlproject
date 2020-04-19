from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
data = np.load('FinalDataSet.npy')
data = data.astype('int')
data = data[:,1:]
numDataPoints = np.size(data, 0)
averageLabel = np.mean(data[:,11])
baselineRMSE = np.sqrt(np.sum(np.square(averageLabel - data[:,11]))/numDataPoints)

# Split data set into training and testing -> 80% training 20% testing
Perc = int(np.around(numDataPoints * 0.7))
trainingData = data[:Perc, 0:11]
testingData = data[Perc:, 0:11]
#print(np.shape(testingData))

# Split labels into training and testing
trainingLabels = data[:Perc, 11]
easierTrainingLabels = np.where(trainingLabels > 5, 1, 0)

testingLabels = data[Perc:, 11]
StandardDev = np.mean(testingLabels)

easierTestingLabels = np.where(testingLabels > 5, 1, 0)
easierStandardDev = np.mean(easierTestingLabels)

# see effect of number of trees in random forest on data
RMSE = np.zeros(100)
accuracy = np.zeros(100)
#RMSEeasy = np.zeros(100)
#accuracyeasy = np.zeros(100)
for i in range(100):
	model = RandomForestClassifier(n_estimators = i+1, 
									bootstrap = True, 
									criterion = 'entropy', 
									max_features = 'sqrt')
	"""
	modelEasy = RandomForestClassifier(n_estimators = i+1, 
									bootstrap = True, 
									criterion = 'entropy', 
									max_features = 'sqrt')
									"""
	model.fit(trainingData, trainingLabels)
	#modelEasy.fit(trainingData, easierTrainingLabels)

	predictedLabels = model.predict(testingData)
	#easierPredLabels = modelEasy.predict(testingData)

	RMSE[i] = np.sqrt(np.sum(np.square(predictedLabels - testingLabels))/np.size(predictedLabels))
	#accuracy[i] = model.score(testingData, testingLabels)
	accuracy[i] = np.sum(np.where(predictedLabels == testingLabels, 1, 0))/np.size(predictedLabels)
accuracy = accuracy * 100
# Optimizing the number of trees in the forest
numberTrees = np.arange(1, 101, 1)
plt.style.use('seaborn')
plt.scatter(numberTrees, accuracy, c = (.26, 0.55, 0.4), s  = 20)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy (%)')
plt.ylim((0, 50))
plt.yticks(np.arange(0, 51, 5))
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, 10))
plt.title('Relationship Between Number of Trees and Accuracy')
plt.tight_layout()
plt.show()
	#RMSEeasy[i] = np.sqrt(np.sum(np.square(easierPredLabels - easierTestingLabels))/np.size(easierPredLabels))
	#accuracyeasy[i] = np.sum(np.where(easierPredLabels == easierTestingLabels, 1, 0))/np.size(easierPredLabels)
	#accuracyeasy[i] = model.score(testingData, easierTestingLabels)
#NRMSE = RMSE/StandardDev
#NRMSEeasy = RMSEeasy/easierStandardDev
#print(accuracy)
"""
numberTrees = np.arange(1, 101, 1)
plt.subplot(1,3,1)
plt.scatter(numberTrees, accuracy, c = 'b')
plt.scatter(numberTrees, accuracyeasy, c = 'r')
plt.subplot(1,3,2)
plt.scatter(numberTrees, RMSE, c = 'b')
plt.scatter(numberTrees, RMSEeasy, c = 'r')
plt.subplot(1,3,3)
plt.scatter(numberTrees, NRMSE, c = 'b')
plt.scatter(numberTrees, NRMSEeasy, c = 'r')
plt.show()
"""



