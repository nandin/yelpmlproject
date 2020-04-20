
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
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


from sklearn.ensemble import AdaBoostRegressor

dt2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                          n_estimators=300, random_state=99)
dt2.fit(trainingX,trainingY)




dt = DecisionTreeRegressor(min_samples_split=20, random_state=99, max_depth = 5)
dt.fit(trainingX, trainingY)

from sklearn.metrics import mean_squared_error, r2_score
dt_score = dt.score(trainingX,trainingY)
#print(dt_score)


from sklearn.metrics import mean_squared_error
from math import sqrt

y_predicted = dt.predict(testingX)
print("Mean squared error: %.2f"% mean_squared_error(testingY, y_predicted))
#print('Test Variance score: %.2f' % r2_score(testingY, y_predicted))

rmse = sqrt(mean_squared_error(testingY, y_predicted))
print(rmse)



#for AdaBoostRegressor
dt2_score = dt2.score(trainingX,trainingY)
y2_predicted = dt2.predict(testingX)
print("Mean squared errorADaboost: %.2f"% mean_squared_error(testingY, y2_predicted))
rmse = sqrt(mean_squared_error(testingY, y2_predicted))
print("AdaBoostRegressor rmse")
print(rmse)



#Unsuccessful Pruning Attempt
'''
from sklearn.tree._tree import TREE_LEAF

def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)

print(sum(dt.tree_.children_left < 0))
# start pruning from the root
prune_index(dt.tree_, 0, 1)
sum(dt.tree_.children_left < 0)
'''


#Unsuccessful Pruning #2
'''
path = dt.cost_complexity_pruning_path(trainingX, trainingY)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
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
