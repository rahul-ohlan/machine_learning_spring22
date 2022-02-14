# 1. neighborClassify Funtion

import numpy as np

def neighborClassify(featureArray, trainArray):

    res = list()

    for height in featureArray:
        min_gap = float('inf')
        min_index = 0
        for i in range(len(trainArray)):
            gap = abs(height-trainArray[i][0])
            if gap < min_gap:
                min_gap = gap
                min_index = i
        
        res.append(int(trainArray[min_index][1]))
    
    return res

# 2. Find Precision Function

def findPrecision(classifierOutput, trueLabels):
    
    n = len(classifierOutput)

    den,num = 0,0
    for i in range(n):
        if classifierOutput[i]== 1:
            den += 1
            if trueLabels[i] == 1:
                num +=1
    
    if den == 0 or num == 0:
        return 0

    return round(num/den,4)


# 3. Remove Blanks function

def removeBlanks(featureArray):

    res = list()

    for i in range(len(featureArray)):

        if featureArray[i][0] == 0 or featureArray[i][1] == 0:
            continue

        res.append(featureArray[i])

    res = np.array(res)
    return res


# featureArray = np.array([6,7,9,-1])
# trainArray = np.array([[0.5,1],[1.5,0],[2.5,0],[4.5,1],[5.5,0],[7.5,1],[8,1],[9.2,1]])

# print(neighborClassify(featureArray,trainArray))

# a = np.array([[0,1],[1,1],[2,3],[1,0]])
# featCorrect = removeBlanks(a)
# print(featCorrect)

# classi = np.array([1,1,0,1,1])
# true = np.array([0,1,1,1,1])
# print(findPrecision(classi,true))