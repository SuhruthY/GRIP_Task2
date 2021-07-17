# Prediction using Decision Tree Algorithm: [Full Code](https://nbviewer.jupyter.org/github/SuhruthY/GRIP_Task2/blob/master/tree.ipynb)
&emsp;The task aims to creating a Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to
predict the right class accordingly.

| Id | Table of Contents                 |
|----|-----------------------------------|
| 1  | Overview                          |
| 2  | Procedure                         |
| 3  | Conclusion                        |
| 4  | References                        |

## Overview
&emsp;The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.  One class is linearly separable from the other 2; the latter are NOT linearly separable from each other. 
- Predicted attribute: class of iris plant.
- Number of Instances: 150 (50 in each of three classes)
- Number of Attributes: 4 numeric, predictive attributes and the class
- Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      - Iris Setosa
      - Iris Versicolour
      - Iris Virginica
       
&emsp;You can find more details here: [Detailed info about Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)
 
&emsp;I used Scikit-learn, a free software machine learning library for the Python programming language to build a Decision Tree Classifier which will then predict the class of Iris correctly. I also used cost complexity pruning to improve the accuracy of my model.

## Procedure
&emsp;We need to now about the *difference between sepal and petal* and some basic knowledge about decision tree. Splitted the data into 2 parts used seperately for training and testing with as test size of 33%. Used the Decision Tree Classifier to predict the class and accuracy_score as a metric. I pruned the final tree to obtain the potential pros and cons of the tree. Python libraries such as Pandas, Numpy, Seaborn, Matplotlib, Sklearn are used.

## Conclusion
&emps;You can experiment with different test sizes or seperate a set for validation. Also try other metrics. You can also tune parameter in various cases neccessary.  

## References
- [Classification: Basic concepts of Decision Tree, and Model Evaluation by Kumar, University of Minnesota](https://www-users.cs.umn.edu/~kumar001/dmbook/ch4.pdf)
- [Decision Tree Classification in Python by Avinash Navlani, datacamp](https://www.datacamp.com/community/tutorials/decision-tree-classification-python)
- [Decision Tree Classifier Documentation by sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Decision Tree on Iris dataset by Chris Edwards, kaggle](https://www.kaggle.com/chrised209/decision-tree-modeling-of-the-iris-dataset)
- [Decision and Classification Trees, Clearly Explained!!! by StatQuest](https://youtu.be/_L39rN6gz7Y)
- [StatQuest: Decision Trees, Part 2 - Feature Selection and Missing Data](https://www.youtube.com/watch?v=7VeUPuFGJHk)



