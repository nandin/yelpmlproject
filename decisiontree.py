
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
import pydotplus
import numpy as np
import collections
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

from sklearn import metrics, tree


img_array = np.load('dataset.npy')
#print(img_array)
#print('\nShape: ',img_array.shape)


dataset = pd.DataFrame({'businessID': img_array[:, 0], 'PostalCode': img_array[:, 1], 'ReviewCount': img_array[:, 2], 'Alcohol': img_array[:, 3], 'Wifi': img_array[:, 4],'GoodForKids': img_array[:, 5],'Delivery': img_array[:, 6], 'Reservations': img_array[:, 7], 'Takeout': img_array[:, 8], 'PriceRange': img_array[:, 9], 'Stars': img_array[:, 10]})

#randomDataset = np.random.rand(len(dataset)) < 0.7

#train = dataset[randomDataset]
#test = dataset[~randomDataset]

from sklearn.model_selection import train_test_split



x = dataset.drop(['Stars','businessID','PostalCode'], axis=1)
y = dataset['Stars']

trainingX = x.iloc[0:1547]
trainingY = y.iloc[0:1547]
testingX = x.iloc[1548:]
testingY = y.iloc[1548:]



#print(x)
#print(y)
#print(train)
#feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

#X = pima[feature_cols] # Features
#y = pima.label # Target variable

##lab_enc = preprocessing.LabelEncoder()
#training_scores_encoded = lab_enc.fit_transform(y)


dt = DecisionTreeRegressor(min_samples_split=20, random_state=99, max_depth = 10)
dt.fit(trainingX, trainingY)

from sklearn.metrics import mean_squared_error, r2_score
dt_score = dt.score(trainingX,trainingY)
print(dt_score)

from sklearn.metrics import mean_squared_error
from math import sqrt

y_predicted = dt.predict(testingX)
print("Mean squared error: %.2f"% mean_squared_error(testingY, y_predicted))
print('Test Variance score: %.2f' % r2_score(testingY, y_predicted))

rmse = sqrt(mean_squared_error(testingY, y_predicted))
print(rmse)


#y_pred = dt.predict(testingX)
#print("Accuracy:",metrics.accuracy_score(testingY, y_pred))
# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
#clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
#clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
#y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#divideArea = int(np.around())



'''
data_feature_names = [ 'ReviewCount', 'Alcohol', 'Wifi','GoodForKids','Delivery','Reservations','Takeout', 'PriceRange']


dot_data = tree.export_graphviz(dt,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


'''





#df = pd.DataFrame(list(img_array.item().iteritems()), columns=['one','two','three','four','five','six','seven', 'eight', 'nine','ten','result'])

#img_array.random.shuffle(x)
#training, test = img_array[:70,:], x[70:,:]
#print(test.shape)
#print(img_array)

# Load the Diabetes dataset
#columns = “one two three four five six seven eight nine ten result”.split() # Declare the columns names
#yelpData = img_array.load_() # Call the diabetes dataset from sklearn
#df = pd.DataFrame(diabetes.data, columns=columns) # load the dataset as a pandas data frame
#y = diabetes.target # define the target variable (dependent variable) as y


# Split into 70% training and 30% testing set
#X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)
