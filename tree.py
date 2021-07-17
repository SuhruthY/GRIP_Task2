import pandas as pd # data analysis and manipulation tool
import numpy as np # large collection of high-level mathematical functions
import seaborn as sns # used for statistical data visualization

import matplotlib.pyplot as plt # used for programmatic plot generation

from sklearn.tree import DecisionTreeClassifier # Loading the classifier
from sklearn.metrics import accuracy_score # laoding the accuracy metric
from sklearn.tree import plot_tree # loading the plot tree function
from sklearn import metrics

# Loading the iris dataset
iris = pd.read_csv("./iris.csv") # a pandas dataframe
iris.set_index("Id", inplace=True) # set the id column as index
iris.head() # first 5 rows

iris.describe() # summary about data

# plotting the iris data
iris.plot(kind="scatter", x="SepalLengthCm",   y="SepalWidthCm") # Make plot of iris dataFrame
plt.show() # display the plot

iris.isnull().any() # checking for null values

sns.pairplot(iris, hue='Species') # Plot pairwise relationships in a dataset
plt.show() # remove extra marking than plot

X = iris.drop("Species", axis=1).values # independent variables
y = iris["Species"] # dependent variable

y_num = y = iris["Species"].replace({"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3 })

from sklearn.model_selection import train_test_split # loading the train_test_split function

X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.33, random_state=42) # splits into train and test 


## Model Building and Evaluation 
tree_clf = DecisionTreeClassifier(random_state=42) # initializing classifier
tree_clf.fit(X_train, y_train) # training the classifier

plt.figure(figsize = (20,10)) # setting the figure size
plot_tree(tree_clf, filled=True) # plotting the initial tree
plt.show() # showing only the plot

y_pred = tree_clf.predict(X_test) # predicting the test data
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # printing the model accuracy

pd.crosstab(y_test, y_pred) ## Misclassification table

# Post Pruning
path = tree_clf.cost_complexity_pruning_path(X_train, y_train) # getting the effective alphas
ccp_alphas, impurities = path.ccp_alphas, path.impurities # seperating alphas to impurities

fig, ax = plt.subplots(figsize=(12,7)) # setting the size and axes

ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post") # plotting alpha vs impurities
ax.set_xlabel("effective alpha") # set the x-label
ax.set_ylabel("total impurity of leaves") # set the y-label
ax.set_title("Total Impurity vs effective alpha for training set") # set the title

plt.show() # Remove unneccesary details

clfs = [] # initializing 
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha) # tree for each alpha
    clf.fit(X_train, y_train) # training for each alpha
    clfs.append(clf) # appending the classifier to the list
print(f"Number of nodes in the last tree is: {clfs[-1].tree_.node_count} with ccp_alpha: {ccp_alphas[-1]}") # print the results

# removing the last classifier
clfs = clfs[:-1] 
ccp_alphas = ccp_alphas[:-1] 

node_counts = [clf.tree_.node_count for clf in clfs] # node counts for each classifier
depth = [clf.tree_.max_depth for clf in clfs] # depth values for each classifier
fig, ax = plt.subplots(2, 1, figsize=(12,7)) # set the axes and create the subplots
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post") # plot aplha vs nodes
ax[0].set_xlabel("alpha") # set x-label for plot-1
ax[0].set_ylabel("number of nodes") # set y-label for plot-1
ax[0].set_title("Number of nodes vs alpha") # set title for plot-1
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post") # plot aplha vs depth
ax[1].set_xlabel("alpha") # set x-label for plot-2
ax[1].set_ylabel("depth of tree") # set y-label for plot-2
ax[1].set_title("Depth vs alpha") # set title for plot-22
fig.tight_layout()

## Accuracy vs Alpha
train_scores = [clf.score(X_train, y_train) for clf in clfs] # train scores for all classifiers
test_scores = [clf.score(X_test, y_test) for clf in clfs] # test scores for all classifiers

fig, ax = plt.subplots(figsize=(12,7)) # set the axes and size
ax.set_xlabel("alpha") # set the x-label
ax.set_ylabel("accuracy") # set the y-label
ax.set_title("Accuracy vs alpha for training and testing sets") # set the title
ax.plot(ccp_alphas, train_scores, marker='o', label="train", 
        drawstyle="steps-post") # plot aplha vs train scores 
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post") # plot alpha vs test scores
ax.legend() # adding legend
plt.show() # excluding all other things