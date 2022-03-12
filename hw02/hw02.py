# import libraries
import pandas as pd
import numpy as np
import time

# load data, prepare training set
df = pd.read_csv("hw2dataNorm.csv")
df=df.iloc[:,1:]
fullData = df.to_numpy()

X = fullData[:,0:10]  # training set
y = fullData[:,10]    # class Labels 

# ****************************************************************************
# Problem 1:

def sigmoidLikelihood(X, y , w):
    
    rows,cols = X.shape
    
    dummy = np.ones(rows)
    dummy = dummy.reshape((rows,1))
    X1 = np.append(X,dummy,axis=1)
    
    def sigmoid(w_vector,xi):
        return (1/(1+ np.exp(-(np.dot(w_vector,xi)))))
    
    def pseudo_probability(w,xi,yi):
        return ((1-sigmoid(w,xi))**(1-yi))*((sigmoid(w,xi))**yi)
    
    vector = list()
    for row in range(len(y)):
        vector.append(pseudo_probability(w,X1[row,:],y[row]))
        
    return np.array(vector) 

# ****************************************************************************
# Problem 2(a):

# using hit and trial

# 0.05**248  --> 2e-323
# # 0.05**249 --> 0.0
# we see that, at 249 multipications of likelihoods, the resultant becomes 0 

# Problem 2(b):

a = np.array([0.05]*249)
res = np.sum(np.log(a))

print('answer to part 2(b): ',res)

# result = -745.9373361149437


# ****************************************************************************
# Problem 3:

# log_Likelihood function returns likelihood of complete input data given the vector w and labels

def log_Likelihood(X, y, W):
    
    rows, cols = X.shape
    
    def sigmoid(w_vector,xi):
        return (1/(1+ np.exp(-(np.sum(w_vector*xi)))))
    
    
    
    def pseudo_probability(w_vector,xi,yi):
        
        yi = int(yi)
        
        f = (1 - sigmoid(w_vector,xi))**(1-yi)
        s = (sigmoid(w_vector,xi))**yi
        
        res = f * s
        return res
    
    
    new_array = list()
    
    for i in range(rows):
        new_array.append(pseudo_probability(W,X[i],y[i]))

    new_array = np.array(new_array)
    log_like = np.sum(np.log(new_array))  
    return log_like

# updateRule to find gradients in order to update vector w

def updateRule(X, y, w):
    
    learning_rate = 0.01
    
    def sigmoid(w_vector,xi):
        sig =  (1/(1+ np.exp(-(np.dot(w_vector,xi)))))

        return sig
    
    
    rows, cols = X.shape
    
    # initialize gradient vector to all zeros
    grad_vector = np.zeros(cols)
    for i in range(rows):
        
        
        for j in range(cols):
            
            xj = X[i][j]
            yi = y[i]
            
            grad_vector[j] += xj*(yi - sigmoid(w,X[i])) # calculating gradients based on all of dataset
            
    
    new_w = w.copy()
    
    for j in range(cols):        # this loop will be removed in the learnlogisticFast function
        new_w[j] += learning_rate*grad_vector[j]

    w = new_w

    return w



def learnLogistic(w0, X, y, K):
    
    log_likelihood = list()
    rows,cols = X.shape
    
    dummy = np.ones(rows)
    dummy = dummy.reshape((rows,1))
    X1 = np.append(X,dummy,axis=1)
    
    
    
    for loop in range(K):
    
    
        w0 = updateRule(X1, y, w0)
        
        log_likelihood.append(log_Likelihood(X1,y,w0))
            
    
    
    
        
    return w0, np.array(log_likelihood)

# Sample Run:

# timeStart = time.time()
# w, LHistory = learnLogistic(w0, X, y, K)
# timeEnd = time.time()
# a = timeEnd-timeStart

# print(w)
# print(LHistory)


# [ 0.31582482 -0.21379867  0.66178671  0.44060446 -0.09422403  0.83404356
#   4.76143471 -1.12802087  1.15673754 -3.63355023  0.45697398]

# [-497.27729395 -463.3764476  -457.44326896 -453.55682843 -450.3679247
#  -447.60265937 -445.19301184 -443.10375312 -441.30709101 -439.77603386]


# ****************************************************************************
# Problem 4:

def updateRuleFast(X, y, w):
    
    learning_rate = 0.01
    
    def sigmoid(w_vector,xi):
        sig =  (1/(1+ np.exp(-(np.dot(w_vector,xi)))))

        return sig
    
    
    rows, cols = X.shape
    
    # initialize gradient vector to all zeros
    grad_vector = np.zeros(cols) # reset updates
    for i in range(rows):
        
        yi = y[i]
            
        grad_vector += X[i]*(yi - sigmoid(w,X[i])) # calculating gradients based on all of dataset
            
    
    
    # using Vectorization to update w skipping the for loop
    
    new_w = w + (learning_rate)*grad_vector
    w = new_w
            
        
    return w


def learnLogisticFast(w0, X, y, K):
    
    log_likelihood = list()
    rows,cols = X.shape
    
    dummy = np.ones(rows)
    dummy = dummy.reshape((rows,1))
    X1 = np.append(X,dummy,axis=1)
    
    
    
    for loop in range(K):
    
    
        w0 = updateRuleFast(X1, y, w0)
        
        log_likelihood.append(log_Likelihood(X1,y,w0))


    return w0, np.array(log_likelihood)

# Sample Output:

# timeStart = time.time()
# w1 = np.zeros(len(X[0])+1)
# wFast, LHistoryFast = learnLogisticFast(w1, X, y, K)
# timeEnd = time.time()
# b = timeEnd-timeStart
# print(wFast)
# print(LHistoryFast)
# print(b)

# [ 0.31582482 -0.21379867  0.66178671  0.44060446 -0.09422403  0.83404356
#   4.76143471 -1.12802087  1.15673754 -3.63355023  0.45697398]


# [-497.27729395 -463.3764476  -457.44326896 -453.55682843 -450.3679247
#  -447.60265937 -445.19301184 -443.10375312 -441.30709101 -439.77603386]


# time difference: *****************************
# print(b-a)

# 0.05612802505493164

# ****************************************************************************
# Problem 5:


def logisticClassify(X, w):
    
    rows,cols = X.shape
    
    dummy = np.ones(rows)
    dummy = dummy.reshape((rows,1))
    X1 = np.append(X,dummy,axis=1)
    
    classLabels = list()
    for i in range(N):
        
        if np.dot(w,X[i]) > 0:
            classLabels.append(1)
        else:
            classLabels.append(0)
            
    return np.array(classLabels)

# classLabels = logisticClassify(X,w) # w as output from learnlogistic

# Sample Output:

# print(classLabels)
# [1 1 1 ... 0 0 1]

# # Accuracy computation:

# c = True 0s
# d = True 1s 

# c,d = 0,0

# for i in range(len(y)):
#     if y[i] == 0 and labels[i] == 0:
#         c+=1

#     elif y[i] == 1 and labels[i] == 1:
#         d +=1

# print('accuracy: ',(c+d)/len(y))

# accuracy:  0.9485


# ****************************************************************************