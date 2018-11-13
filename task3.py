import csv

import numpy as np
import pandas as pd

# Minimum number of observations for an sample
from biosppy.signals import ecg
from sklearn import svm

import utils

# utils.produce_final_x_train()
# utils.produce_final_y_train()
# exit()


df_Xtrain = pd.read_csv("data/X_trainO_final.csv", delimiter=",", index_col="id", dtype=np.float64)
df_ytrain = pd.read_csv("data/y_trainO_final.csv", delimiter=",", index_col="id", dtype=int)
# Creates and trains model
SVM_clf = svm.SVC(decision_function_shape='ovo')
SVM_clf.fit(df_Xtrain, df_ytrain)
# df_Xtest = pd.read_csv("data/X_test_final.csv", delimiter=",", index_col="id", dtype=np.float64)
y_preds = []
with open('data/X_test.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader, None)
    for row in spamreader:
        row = np.array(row[1:], dtype=int)
        features = utils.transform_line(row, len(row))
        if features:
            features = np.array(features).reshape(1, -1)
            y_preds.append(SVM_clf.predict(features))
        else:
            print("error")
print(y_preds)
exit()

# Loading y_train.csv into panda Dataframe
y_train = np.genfromtxt("data/y_train.csv", delimiter=",", skip_header=1)
y_train = np.delete(y_train, 0, 1)  # deletes first colums ->id
y_train = y_train.astype(np.float)
df_ytrain = pd.DataFrame(data=y_train)

# Loading X_test.csv into panda DataFrame
X_test = np.genfromtxt("data/X_test.csv", delimiter=",", skip_header=1)
X_test = np.delete(X_test, 0, 1)
X_test = X_test.astype(np.float)
df_Xtest = pd.DataFrame(data=X_test)

# Writing the predictions to a CSV
"""y_pred = pd.DataFrame(y_pred, columns=["y"])
indexes = np.arange(0, len(y_pred))
y_pred.index = indexes.astype(np.float)
y_pred.index.names = ['id']
output_name = 'predictions.csv'
y_pred.to_csv(output_name)"""
