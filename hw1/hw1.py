import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

# import data
mat = scipy.io.loadmat('hw1data.mat')
trainData = mat['trainData']
testData = mat['testData']

# **********************************************************************************************
# Question 1
# inspect distributions of test data for each class

# step 1: get data for each of the classes

# 0 : Cairo
# 1 : Frankfurt
# 2 : Philadelphia
# 3 : Seoul

cairo = list()
ffurt = list()
phily = list()
seoul = list()

for city in testData:
    if city[0] == 0:
        cairo.append(city[1])
    elif city[0] == 1:
        ffurt.append(city[1])
    elif city[0] == 2:
        phily.append(city[1])
    else:
        seoul.append(city[1])

# step 2 : plot and inspect histograms for each class

# cairo distribution

plt.hist(cairo,edgecolor = 'black')
plt.xlabel('postTimes')
plt.ylabel('frequency')
plt.title('postTimes distribution : Cairo')
plt.show()

# frankfurt distribution

plt.hist(ffurt,edgecolor = 'black')
plt.xlabel('postTimes')
plt.ylabel('frequency')
plt.title('postTimes distribution : Frankfurt')
plt.show()

# philadelphia distribution

plt.hist(phily,edgecolor = 'black')
plt.xlabel('postTimes')
plt.ylabel('frequency')
plt.title('postTimes distribution : Philadelphia')
plt.show()

# seoul distribution

plt.hist(seoul,edgecolor = 'black')
plt.xlabel('postTimes')
plt.ylabel('frequency')
plt.title('postTimes distribution : Seoul')
plt.show()

# Step 3: report results

# class   city               distribution
#   0    Cairo           Gaussian Distribution
#   1    Frankfurt       Gaussian Distribution
#   2    Philadelphia    Gaussian Distribution
#   3    Seoul           Gaussian Distribution

 
# **********************************************************************************************
 # Question 2:
#learnParams function
# returns mean and standard deviation of input data

def learnParams(Data):
    d = dict()

    for i in Data:

        if i[0] not in d:
            d[i[0]] = []
            d[i[0]].append(i[1])
            continue

        d[i[0]].append(i[1])

    target_variable = sorted([i for i,v in d.items()])

    res = list()
    for i in target_variable:
        d[i] = np.array(d[i])
        mean = np.mean(d[i])
        standard_deviation = np.std(d[i])
        res.append([mean,standard_deviation])

    res = np.array(res)

    return res


# **********************************************************************************************
# Question 3:
# learnPriors function
# returns prior probabilities of each class

def learnPriors(Data):

    d = dict()
    total = len(Data)

    for i in Data:

        d[i[0]] = d.get(i[0],0) + 1

    classes = sorted([i for i,v in d.items()])
    res = list()
    for clas in classes:

        res.append([d[clas]/total])

    res = np.array(res)

    return res        

# **********************************************************************************************
# Question 4:
# labelBayes function
# returns classification of each class from input data

def normal_dist(mu,sd,x): # for calculating likelihoods

    proby = (1/(sd*np.sqrt(2*np.pi)))*np.exp((-0.5)*((x-mu)/sd)**2)

    return proby

def labelBayes(postTimes,paramsL,priors):

    

    # class labels are indices of paramsL
    res = list()
    for tim in postTimes:
        
        default_post = float('-inf')
        default_class = 0
        for clas in range(len(paramsL)):
            mu = paramsL[clas,0]
            sd = paramsL[clas,1]
            prior = priors[clas]
            likelihood = normal_dist(mu,sd,tim)
            posterior = likelihood * prior
            if posterior > default_post:
                default_post = posterior
                default_class = clas
        res.append(default_class)
    
    res = np.array(res)
    return res


# **********************************************************************************************
# Question 5:
# evaluateBayes function
# returns accuracy of classification for input data against given labels

def evaluateBayes(paramsL,priors,testData):

    postTimes = testData[:,1]

    classes = labelBayes(postTimes,paramsL,priors)

    den = len(testData)
    num = 0
    for i in range(len(testData)):

        if testData[i,0] == classes[i]:
            num += 1

    result = num/den

    return result


# **********************************************************************************************
# Question 6:
# learnParamsClock function
# modified learnParams function that more reflects the circular nature of clock on input data

def learnParamsClock(Data):

    # if original postTimes are in range: 0-59 100-159 200-259 300-359 ... 2300-2359
    # we update the postTimes to ranges : 0-59 60-119  120-179 180-239 ... 1380-1439
    # by reducing by the difference :      0     40      80      120   ... 920
    # now the data represents abolute minute of the day instead of hhmm integer format
    # THIS MAKES THE DATA CONTINUOUS as now it is in the range 0 1 2 3 .... 1439
    # we just modify the data pass it on as input to the original learnParamsfunction



    minuteMap = dict()

    for value in Data[:,1]:
        hours = value//100
        minuteMap[value] = value - hours*40
    
    modified_data = list()

    for val in Data:
        modified_data.append([val[0],minuteMap[val[1]]])

    modified_data = np.array(modified_data)

    return learnParams(modified_data)