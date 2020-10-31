#~ various sources I used to learn how to code the perceptron

#https://www.youtube.com/watch?v=tA9jlwXglng
#https://arxiv.org/abs/1903.08519
#https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/  -- this was an amazing tutorial page
#https://julienbeaulieu.gitbook.io/wiki/sciences/machine-learning/neural-networks/perceptron-algorithm
#https://queirozf.com/entries/add-labels-and-text-to-matplotlib-plots-annotation-examples
#https://likegeeks.com/numpy-where-tutorial/
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
#https://www.kite.com/python/answers/how-to-plot-a-line-of-best-fit-in-python


# I understood the concept of a perceptron and how the features are affected
# by the weights to find the model based on the dataset
# however I struggled with executing the code on the perceptron
# I spent all week watching tutorial videos and different coding samples
# and still had trouble calculating the training data



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Perceptron learning algorithm

def train_data(train, data):
    weights = [-.2, .54]
    for run in range(100):
        sum_error = 0.00 #reset at beginning of each loop
        for row in train:
            prediction = predict(row, weights)
            sum_error += (row[-1] - prediction)**2 #subtracts answer by predicted value
            weights[0] = weights[0] + data * (row[-1] - prediction)
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + data * (row[-1] - prediction) * row[i]
        print('>runs=%d, error=%.3f' % (run, sum_error))
    return weights

#to find the predicted value with weights applied
def predict(row, weight):
    guess = weight[0]
    for i in range(len(row)-1):
        guess += weight[i+1] * row[i]
        if(guess >= 0):
            value = 1
        else:
            value = 0
    return value #gives us pediction value with given weight


#sample data set used in many perceptron algorithms
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
weight = [-.2, .54, .04, -.7]

#read in values to X and y in order to determine their linear dependence
y = data.iloc[0:100,4].values
y = np.where(y == 'Iris-sertosa',-1,1 )   #changes label of iris setosa to int '1'

X = data.iloc[0:100, [0,2]].values

for row in X:
    prediction = predict(row, weight)
    print("Expected=%d, Predicted=%d" % (row[-1], prediction))
weights = train_data(X, 1)
print(weights)

#creating the scatter plot
#We can see from looking at the data set that the first 50 are setosa and the next 50 are versicolor
plt.scatter(X[:50,0],X[:50, 1], color = 'blue', marker = 'o', label = 'setosa') #first 50 rows in first and second column,
plt.scatter(X[50:,0],X[50:, 1], color = 'red', marker = 'o', label = 'versicolor') #50 to the end rows in first and second column,
plt.xlabel('sepal_length')
plt.ylabel('petal_length')
plt.text(6,2.5,'data appears linearly separable')

m, b = np.polyfit(X[:50,0], X[:50, 1], 1)
plt.plot(X[:50,0], m*X[:50,0] + b)

m, b = np.polyfit(X[50:,0], X[50:, 1], 1)
plt.plot(X[50:,0], m*X[50:,0] + b)

plt.show()


error = train_data(X[0:], 100)
num_Of_Runs = np.arange(1, 100)
