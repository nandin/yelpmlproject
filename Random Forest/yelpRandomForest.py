from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
data = np.load('FinalDataSet.npy')
data = data[~pd.isnull(data).any(axis=1)]
data = data.astype('int')
data = data[:,1:]
numDataPoints = np.size(data, 0)

# Split data set into training and testing -> 80% training 20% testing
Perc = int(np.around(numDataPoints * 0.7))
trainingData = data[:Perc, 0:11]
testingData = data[Perc:, 0:11]
#print(np.shape(testingData))

# Split labels into training and testing
trainingLabels = data[:Perc, 11]
easierTrainingLabels = np.where(trainingLabels > 5, 1, 0)

testingLabels = data[Perc:, 11]
easierTestingLabels = np.where(testingLabels > 5, 1, 0)

# see effect of number of trees in random forest on data
accuracy = np.zeros(100)
accuracyEasy = np.zeros(100)


# Optimizing the number of trees in the forest
for i in range(100):
	model = RandomForestClassifier(n_estimators = i + 1, 
									bootstrap = True, 
									criterion = 'entropy', 
									max_features = 'sqrt')
	
	modelEasy= RandomForestClassifier(n_estimators = i + 1, 
									bootstrap = True, 
									criterion = 'entropy', 
									max_features = 'sqrt')

	model.fit(trainingData, trainingLabels)
	modelEasy.fit(trainingData, easierTrainingLabels)
	
	accuracy[i] = model.score(testingData, testingLabels) * 100
	accuracyEasy[i] = modelEasy.score(testingData, easierTestingLabels) * 100

numberTrees = np.arange(1, 101, 1)
plt.style.use('seaborn')
MultiClass = plt.scatter(numberTrees, accuracy, c = (0.27, 0.58, 0.37), s  = 30)
BinaryClass = plt.scatter(numberTrees, accuracyEasy, c = (.15, 0.35, 0.58), s = 30)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy (%)')
plt.ylim((0, 100))
plt.yticks(np.arange(0, 101, 10))
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, 10))
#plt.title('Relationship Between Number of Trees and Accuracy')
plt.title('Accuracy Difference Between Binary and Multi-Class Labels')
plt.legend((MultiClass, BinaryClass),
           ('Multi-Class Label', 'Binary Label'))
plt.tight_layout()
plt.show()
"""
# Difference between split criterion
accuracyEntropy = np.zeros(100)
accuracyGini = np.zeros(100)
numberTrees = np.arange(1, 101, 1)
for i in range(100):
	modelEntropy = RandomForestClassifier(n_estimators = i + 1,
									bootstrap = True,
									criterion = 'entropy',
									max_features = 'sqrt')
	modelGini = RandomForestClassifier(n_estimators = i + 1,
									bootstrap = True,
									criterion = 'gini',
									max_features = 'sqrt')
	modelEntropy.fit(trainingData, trainingLabels)
	modelGini.fit(trainingData, trainingLabels)
	accuracyEntropy[i] = modelEntropy.score(testingData, testingLabels) * 100
	accuracyGini[i] = modelGini.score(testingData, testingLabels) * 100
plt.style.use('seaborn')
Entropy = plt.scatter(numberTrees, accuracyEntropy, c = (.15, 0.35, 0.58), s  = 30)
Gini = plt.scatter(numberTrees, accuracyGini, c = (0.27, 0.58, 0.37), s = 30)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy (%)')
plt.title('Effect of Split Criterion')
plt.legend((Entropy, Gini),
           ('Information Gain', 'Gini Impurity'))
plt.ylim((20, 30))
plt.yticks(np.arange(20, 31, 1))
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, 10))
plt.tight_layout()
plt.show()


# Max Depth vs Accuracy

accuracyDepth = np.zeros(90)
depth = np.arange(10, 100, 1)
for i in range(90):
	modelDepth = RandomForestClassifier(n_estimators = 30,
										bootstrap = True,
										criterion = 'entropy',
										max_features = 'sqrt',
										max_depth = depth[i])
	modelDepth.fit(trainingData, trainingLabels)
	accuracyDepth[i] = modelDepth.score(testingData, testingLabels) * 100
plt.style.use('seaborn')
plt.scatter(depth, accuracyDepth, c = (0.27, 0.58, 0.37), s  = 30)
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy (%)')
plt.title('Maximum Tree Depth vs Accuracy')
plt.ylim((24, 31))
plt.yticks(np.arange(24, 35, 1))
plt.xlim((10, 100))
plt.xticks(np.arange(10, 101, 10))
plt.tight_layout()
plt.show()
	#RMSEeasy[i] = np.sqrt(np.sum(np.square(easierPredLabels - easierTestingLabels))/np.size(easierPredLabels))
	#accuracyeasy[i] = np.sum(np.where(easierPredLabels == easierTestingLabels, 1, 0))/np.size(easierPredLabels)
	#accuracyeasy[i] = model.score(testingData, easierTestingLabels)
#NRMSE = RMSE/StandardDev
#NRMSEeasy = RMSEeasy/easierStandardDev
#print(accuracy)


# Run with optimized values
modelOpt = RandomForestClassifier(n_estimators = 30,
									bootstrap = True,
									criterion = 'entropy',
									max_features = 'sqrt',
									max_depth = 10)
modelOpt.fit(trainingData, trainingLabels)
accuracyOpt = modelOpt.score(testingData, testingLabels) * 100
print(accuracyOpt)
"""

