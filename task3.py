import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg

# Minimum number of observations for an sample
selected_threshold = 8600

# Undersample data
#utils.save_undersampled_data(selected_threshold)

# Loadinf X_train.csv into panda Dataframe
df_Xtrain = pd.read_csv("data/X_train_undersampled.csv", delimiter=",", index_col="id", dtype=int)
extracted_features = []
i = 0
for signal in df_Xtrain.values:
    if len(ecg.christov_segmenter(signal, 300)[0]) == 0:
        print(i)
    i += 1
exit()
for signal in df_Xtrain.values:
    ts, filtered, r_peaks_indices, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal, 300, False)
    # R-peaks
    r_peaks_count = len(r_peaks_indices)
    r_peaks = list(map(lambda i: signal[i], r_peaks_indices))
    max_r_peak = max(r_peaks)
    min_r_peak = min(r_peaks)
    avg_r_peak = np.average(r_peaks)
    r_peaks_frequency = r_peaks_count / (selected_threshold / 300)
    # Heart rate
    # min_bpm = min(heart_rate)
    # max_bpm = max(heart_rate)
    avg_bpm = sum(heart_rate) / len(heart_rate)
    std_bpm = np.std(heart_rate)
    feature = [r_peaks_count, r_peaks_frequency, max_r_peak, min_r_peak, avg_r_peak, avg_bpm, std_bpm]
    extracted_features.append(feature)

# DataFrame creation
df = pd.DataFrame(np.array(extracted_features),
                  index=df_Xtrain["id"],
                  columns=["r_peaks_count", "r_peaks_frequency", "r_peaks_max", "r_peaks_min",
                           "r_peaks_avg", "bpm_avg", "bpm_std"])
df.to_csv("data/X_train_final")

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
