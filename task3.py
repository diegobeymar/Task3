from sklearn.metrics import r2_score
from sklearn import preprocessing
from statistics import mean
from csv import reader
import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

#Loadinf X_train.csv into panda Dataframe
X_train = np.genfromtxt("X_train.csv", delimiter = ",", skip_header = 1)
X_train = np.delete(X_train,0,1)
X_train = X_train.astype(np.float)
df_Xtrain = pd.DataFrame(data = X_train)

#Loading y_train.csv into panda Dataframe
y_train = np.genfromtxt("y_train.csv", delimiter = ",", skip_header = 1)
y_train = np.delete(y_train,0,1)#deletes first colums ->id
y_train = y_train.astype(np.float)
df_ytrain = pd.DataFrame(data = y_train)

#Loading X_test.csv into panda DataFrame
X_test = np.genfromtxt("X_test.csv", delimiter = ",", skip_header = 1)
X_test = np.delete(X_test,0,1)
X_test = X_test.astype(np.float)
df_Xtest = pd.DataFrame(data = X_test)



#   UNDERSAMPLING
X_trainU = X_train
y_trainU = y_train
indexes = []
#GETTING INDEXES OF CLASS 1 #########
for index in range(4800):
    if y_trainU[index] == 1.0:
        indexes.append(index)
#####################################

print("indexes size ",len(indexes))
indexes3000 = indexes[:3000]
print("size indexes3000: ",len(indexes3000))

y_trainU = np.delete(y_trainU,indexes3000,0)
X_trainU = np.delete(X_trainU,indexes3000,0)
print("y_trainU size: ", y_trainU.shape)
print("X_trainU size: ", X_trainU.shape)


#   OVERSAMPLING
X_trainO = X_train
y_trainO = y_train
toCopy = []
#Getting indexes of classes 0 and 2
for index in range(4800):
    if y_trainO[index] == 0.0 or y_trainO[index] == 2.0:
        toCopy.append(index)
########################################
#np.asarray(toCopy)
#np.repeat(X_trainO, repeats = toCopy, axis=0)
TwoClassesCopy = X_trainO[toCopy,:].copy()
TwoClassesCopyY = y_trainO[toCopy,:].copy()
X_trainO = np.concatenate((X_trainO, TwoClassesCopy, TwoClassesCopy, TwoClassesCopy, TwoClassesCopy, TwoClassesCopy),axis = 0)
y_trainO = np.concatenate((y_trainO, TwoClassesCopyY, TwoClassesCopyY, TwoClassesCopyY, TwoClassesCopyY, TwoClassesCopyY),axis = 0)
print(X_trainO.shape)
print(y_trainO.shape)

###############RANDOM FOREST CLASSIFIER UNBALANCED DATASET ###################

"""clf = RandomForestClassifier(n_estimators = 500, max_depth = 12, random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict(df_Xtest)

#Writing the predictions to a CSV
y_predP = pd.DataFrame(y_pred, columns=["y"])
indexes = np.arange(0, len(y_predP))
y_predP.index = indexes.astype(np.int)
y_predP.index.names = ['id']
output_name = 'predictions.csv'
y_predP.to_csv(output_name)
print("y_predP shape: ", y_predP.shape)"""

###############RANDOM FOREST CLASSIFIER OVERSAMPLING ####################
"""clfO = RandomForestClassifier(n_estimators = 500, max_depth = 12, random_state = 0)
clfO.fit(X_trainO, y_trainO)
y_predO = clfO.predict(df_Xtest)

#Writing the predictions to a CSV
y_predPO = pd.DataFrame(y_predO, columns=["y"])
indexes = np.arange(0, len(y_predPO))
y_predPO.index = indexes.astype(np.int)
y_predPO.index.names = ['id']
output_name = 'predictionsO.csv'
y_predPO.to_csv(output_name)
print("y_predPO shape: ", y_predPO.shape)"""
###############RANDOM FOREST CLASSIFIER UNDERSAMPLING####################

clfU = RandomForestClassifier(n_estimators = 500, max_depth = 12, random_state = 0)
"""clfU.fit(X_trainU, y_trainU)

y_predU = clfU.predict(df_Xtest)

#Writing the predictions to a CSV
y_predPU = pd.DataFrame(y_predU, columns=["y"])
indexes = np.arange(0, len(y_predPU))
y_predPU.index = indexes.astype(np.int)
y_predPU.index.names = ['id']
output_name = 'predictionsU.csv'
y_predPU.to_csv(output_name)
print("y_predPU shape: ", y_predPU.shape)
#print("train dataset score: ", clf.score(X_trainU,y_trainU))"""

#################SUPPORT VECTOR MACHINES############################

SVM_clf = svm.SVC(decision_function_shape='ovo')
SVM_clf.fit(X_trainO, y_trainO)
#SVM_clf.fit(df_Xtrain,df_ytrain)
y_predSVM = SVM_clf.predict(df_Xtest)

y_predPSVM = pd.DataFrame(y_predSVM, columns=["y"])
indexes = np.arange(0, len(y_predPSVM))
y_predPSVM.index = indexes.astype(np.int)
y_predPSVM.index.names = ['id']
output_name = 'predictionsSVMOvere.csv'
y_predPSVM.to_csv(output_name)

#################VOTING WITH SKLEAR IN BUILT FUNCTION
"""from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

clf3 = GaussianNB()

eclf1 = VotingClassifier(estimators=[('rf',clfU),('svm',SVM_clf)], voting='hard')
eclf1.fit(X_trainU,y_trainU)
y_predVote = eclf1.predict(df_Xtest)

y_predPVote = pd.DataFrame(y_predVote, columns=["y"])
indexes = np.arange(0, len(y_predPVote))
y_predPVote.index = indexes.astype(np.int)
y_predPVote.index.names = ['id']
output_name = 'predictionsVote.csv'
y_predPVote.to_csv(output_name)"""

#################VOTING OF THE THREE RANDOM FOREST CLASSIFIERS#####################

#print("S: ", y_predP.at[0,'y']," U: ", y_predPU.at[0,'y'], " O: ", y_predPO.at[0,'y'])
"""y_predV = []
for i in range(4100):
    if y_predP.at[i,'y'] == y_predPU.at[i,'y'] and y_predP.at[i,'y'] == y_predPO.at[i,'y']:
        y_predV.append(y_predP.at[i,'y'])
    elif y_predP.at[i,'y'] == y_predPU.at[i,'y'] and y_predPU.at[i,'y'] == y_predPO.at[i,'y']:
        y_predV.append(y_predPU.at[i,'y'])
    elif y_predP.at[i,'y'] == y_predPO.at[i,'y'] and y_predPU.at[i,'y'] == y_predPO.at[i,'y']:
        y_predV.append(y_predPO.at[i,'y'])
    else:
        y_predV.append(y_predPU.at[i,'y'])

y_predV = np.asarray(y_predV)
y_predV = np.transpose(y_predV)
print("y_predV shape: ", y_predV.shape)

#Writing the predictions to a CSV
y_predPV = pd.DataFrame(y_predV, columns=["y"])
indexes = np.arange(0, len(y_predPV))
y_predPV.index = indexes.astype(np.int)
y_predPV.index.names = ['id']
output_name = 'predictionsV.csv'
y_predPV.to_csv(output_name)"""

#LinearRegression

"""print("Regression")
reg = LinearRegression().fit(df_Xtrain,df_ytrain)
train_score = reg.score(df_Xtrain,df_ytrain)
print(train_score)


#random forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000,max_features = 0.525, random_state = 42)
rf.fit(df_Xtrain, df_ytrain)
print("forest score ", rf.score(df_Xtrain, df_ytrain))

#print("train score = ", train_score)

print("Prediction")
#y_pred = reg.predict(df_Xtest)

y_pred = rf.predict(df_Xtest)

#score = r2_score(y_train, y_pred)
#print("score = ", score)

#Writing the predictions to a CSV
y_predP = pd.DataFrame(y_pred, columns=["y"])
indexes = np.arange(0, len(y_predP))
y_predP.index = indexes.astype(np.float)
y_predP.index.names = ['id']
output_name = 'predictions.csv'
y_predP.to_csv(output_name)"""

On this task I first tried to apply undersampling and then oversampling, the last method gave a better public score. It only multiplies the instances of the second class by 2.
My code first loads the CSV files, put them into arrays and iterates over the rows, of the training dataset. If this row is class 0 or 2, it is copied to an array that will be added twice to the dataset, in order to obtain the same number of rows for each class. Finally, I apply the support vector machines algorithm to this new dataset, with the decision function shape "one vs one", since it is the one that gave better results.
Before applying this method I tried random forest and Bayes, I tried to apply many estimators and make them vote for the best solution, but the public score obtained was inferior to the one with the support vector machines and oversampled training dataset.
