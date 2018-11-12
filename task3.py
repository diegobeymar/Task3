import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg

# Minimum number of observations for an sample
selected_threshold = 8600

# Undersample data
# utils.save_undersampled_data(selected_threshold)

# Loadinf X_train.csv into panda Dataframe
df_Xtrain = pd.read_csv("data/X_train_undersampled.csv", delimiter=",", index_col="id")

print(ecg.christov_segmenter(df_Xtrain.iloc[0], sampling_rate=300))
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
