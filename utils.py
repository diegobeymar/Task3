import csv
from functools import reduce

import pandas as pd

import numpy as np
from biosppy.signals import ecg


def get_stats(thresholds):
    # Min length of row: 2606
    # Percentage over 8600 samples: 0.8112543962485346
    # Percentage over 8630 samples: 0.7667057444314185
    # Percentage over 8640 samples: 0.7508792497069168
    # Percentage over 8660 samples: 0.7237202032043767
    for threshold in thresholds:
        count = 0
        total = 0
        with open('data/X_train.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                total += 1
                if len(row) > threshold:
                    count += 1
        print("Percentage over " + str(threshold) + " samples: " + str(count / total))


def produce_final_y_train():
    df_Xtrain = pd.read_csv("data/X_train_final.csv", delimiter=",", index_col="id", dtype=np.float64)
    y_train = pd.read_csv("data/y_train.csv", delimiter=",", index_col="id", dtype=int)
    y_train_filtered = y_train.iloc[df_Xtrain.index]
    y_train_filtered.to_csv("data/y_train_final.csv")


def transform_line(signal, nbr_samples):
    ts, filtered, r_peaks_indices, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal, 300, False)
    if len(heart_rate) == 0:
        return None
    # R-peaks
    r_peaks_count = len(r_peaks_indices)
    r_peaks = list(map(lambda i: signal[i], r_peaks_indices))
    max_r_peak = max(r_peaks)
    min_r_peak = min(r_peaks)
    avg_r_peak = np.average(r_peaks)
    r_peaks_frequency = r_peaks_count / (nbr_samples / 300)
    # Heart rate
    # min_bpm = min(heart_rate)
    # max_bpm = max(heart_rate)
    avg_bpm = sum(heart_rate) / len(heart_rate)
    std_bpm = np.std(heart_rate)
    return [r_peaks_count, r_peaks_frequency, max_r_peak, min_r_peak, avg_r_peak, avg_bpm, std_bpm]


def produce_final_x_train():
    extracted_features = []
    with open('data/X_train.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader, None)
        for row in spamreader:
            index = row[0]
            row = np.array(row[1:], dtype=int)
            features = transform_line(row, len(row))
            if features:
                extracted_features.append([index] + features)
            else:
                print("Can't calculate heart beat for x train datapoint")

    # DataFrame creation
    arr = np.array(extracted_features)
    df = pd.DataFrame(arr,
                      columns=["id", "r_peaks_count", "r_peaks_frequency", "r_peaks_max", "r_peaks_min",
                               "r_peaks_avg", "bpm_avg", "bpm_std"])
    df.set_index("id", inplace=True)
    df.to_csv("data/X_train_final.csv")


def localizeClass(labels, clazz):
    indexes = []
    for index in range(len(labels)):
        if labels.values[index] == clazz:
            indexes.append(index)
    return indexes


def oversampling(df_Xtrain, df_ytrain, clazz,nbReplications):
    indexes = localizeClass(df_ytrain, clazz)
    valsX = df_Xtrain.values[indexes,:].copy()
    valsY = df_ytrain.values[indexes,:].copy()
    """for i in range(nbReplications):
        oversampledX.append(valsX)  #  np.concatenate((oversampledX, valsX), axis=0)
        oversampledY.append(valsY)  # np.concatenate((oversampledY, valsX), axis=0)"""
    oversampledX = np.repeat(valsX,nbReplications,axis =0)
    oversampledY = np.repeat(valsY,nbReplications,axis =0)
    return oversampledX, oversampledY


def generate_oversampled_files():
    # Loadinf X_train.csv into panda Dataframe
    df_Xtrain = pd.read_csv("data/X_train_final.csv", delimiter=",", index_col="id", dtype=np.float64)
    df_ytrain = pd.read_csv("data/y_train_final.csv", delimiter=",", index_col="id", dtype=int)

    # Oversampling
    indexesClass1 = localizeClass(df_ytrain, 1)
    print(indexesClass1)
    oversampledX_class0, oversampledY_class0 = oversampling(df_Xtrain, df_ytrain, 0, 1)
    oversampledX_class1, oversampledY_class1 = oversampling(df_Xtrain, df_ytrain, 1, 7)
    oversampledX_class2, oversampledY_class2 = oversampling(df_Xtrain, df_ytrain, 2, 2)
    oversampledX_class3, oversampledY_class3 = oversampling(df_Xtrain, df_ytrain, 3, 18)
    X_train = np.concatenate((oversampledX_class0, oversampledX_class1, oversampledX_class2, oversampledX_class3),
                             axis=0)
    y_train = np.concatenate((oversampledY_class0, oversampledY_class1, oversampledY_class2, oversampledY_class3),
                             axis=0)

    df = pd.DataFrame(X_train,
                      columns=["r_peaks_count", "r_peaks_frequency", "r_peaks_max", "r_peaks_min",
                               "r_peaks_avg", "bpm_avg", "bpm_std"])
    # df.set_index("id", inplace=True)
    df.to_csv("data/X_trainO_final.csv")
    df = pd.DataFrame(y_train,
                      columns=["y"])
    df.to_csv("data/y_trainO_final.csv")
