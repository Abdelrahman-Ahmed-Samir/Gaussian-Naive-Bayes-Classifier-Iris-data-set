# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 00:54:22 2022

@author: Abdelrahman-Ahmed-Samir
"""




from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import random

iris=load_iris()
data_features = iris.data
data_target = iris.target



print('')
print('Calculating....')

"""
train set = 80%
            combine set = 20%
            /         \
           /           \
     test set(10%)     validate set(10%)

"""

def train_validate_test_split (data, labels, testRatio , valRatio):
    length_datatr = len(data)*(1-(testRatio+valRatio))   # X Train set 120 ( rest of data features = 30)
    length_datatrain = int(length_datatr)
    data_train = random.choices(data,k=length_datatrain)
    
    
    length_datate = len(data)- length_datatrain       # X  Test set () (rest of data features = 0)
    length_datatest = int(length_datate)
    data_combine = random.choices(data,k=length_datatest)  #30
    
    length_targett = len(labels)*(1-(testRatio+valRatio)) # Y Train set
    length_targettrain = int(length_targett)
    target_train = random.choices(labels,k=length_targettrain) # 120
    
    length_targette = len(labels)- length_targettrain   # Y Test set
    length_targettest = int(length_targette)
    target_combine = random.choices(labels,k=length_targettest) #30
    # valid and test sets after the split of the combine set!
    # 30 
    length_xvali = len(data_combine)*(0.5)   #15
    length_xvalid = int(length_xvali)
    x_valid = random.choices(data_combine,k=length_xvalid)
    
    length_xtrainn = len(data_combine)-(length_xvalid)
    length_xtrain = int(length_xtrainn)
    x_test = random.choices(data_combine,k=length_xtrain)
    
    length_yvali = len(target_combine)*(0.5)  # 15
    length_yvalid = int(length_yvali)
    y_valid = random.choices(target_combine,k=length_yvalid)
    
    length_ytrainn = len(target_combine)-(length_yvalid)
    length_ytrain = int(length_ytrainn)
    y_test = random.choices(target_combine,k=length_ytrain)
    
    
    
    
    
    gnb = GaussianNB()
    gnb.fit(data_train, target_train)
    target_pred = gnb.predict(data_combine)
    return target_pred , target_combine
    
  
    
    # end
    #calculate accuracy function
def calculate_accuracy(target_pred,target_combine):
     count = 0
     for i in range(len(target_combine)): #30
         if target_pred[i] == target_combine[i]:
             count += 1
     return count/len(target_combine) * 100    
 
    



prediction_y , y = train_validate_test_split(data_features,data_target,0.3,0.3)


    

print(calculate_accuracy(prediction_y, y))


                            #Decision Boundary
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_decision_boundaries(X, y, model_class, **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting 
    the model as we need to find the predicted value for every point in 
    scatter plot.
    Arguments:
            X: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model_class: A Scikit-learn ML estimator class 
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            LogisticRegression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator
    
    Typical code example:
            plt.figure()
            plt.title("KNN decision boundary with neighbros: 5",fontsize=16)
            plot_decision_boundaries(X_train,y_train,KNeighborsClassifier,n_neighbors=5)
            plt.show()
    """
    try:
        X = np.array(X)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")
    # Reduces to the first two columns of data
    reduced_data = X[:, :2]
    # Instantiate the model object
    model = model_class(**model_params)
    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].    

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Sepal Length",fontsize=15)
    plt.ylabel("Sepal Width",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    #plotting text labels in the left bottom
    # Define axes

    left = 0.01
    width = 0.9
    bottom  = 0.01
    height = 0.9
    right = left + width
    top = bottom + height
    ax = plt.gca()

    # Transform axes

    ax.set_transform(ax.transAxes)

    # Define text


    ax.text(left, bottom, """                            purple:setosa 
                            green:versicolor
                            yellow:virginica""",
    horizontalalignment='left',
    verticalalignment='bottom',
    color='r',size=10,
    transform=ax.transAxes)
    
    # Display Graph

    plt.show()
    
    return plt



print("Decision Boundaries are up there ^ ^ ^ ")

plt.figure()

plt.title("Naive Bayes decision boundary", fontsize=16)
plot_decision_boundaries(data_features, data_target, GaussianNB)
plt.show()


#---------------------------------Graph (Sepal)--------------------------------


# The indices of the features that we are plotting (class 0 & 1)
x_index = 0
y_index = 1
# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()
plt.show()


#------------------------------------Graph (petal)-----------------------------

# The indices of the features that we are plotting (class 2 & 3)
x_index = 2
y_index = 3
# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------



