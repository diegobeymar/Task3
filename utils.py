import csv
import pandas as pd

import numpy as np


def get_stats(thresholds):
    # Min length of row: 2606
    # Percentage over 8600 samples: 0.8112543962485346
    # Percentage over 8630 samples: 0.7667057444314185
    # Percentage over 8640 samples: 0.7508792497069168
    # Percentage over 8660 samples: 0.7237202032043767
    for threshold in thresholds:
        data = []
        count = 0
        total = 0
        with open('data/X_train.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                total += 1
                if len(row) > threshold:
                    count += 1
        print("Percentage over " + str(threshold) + " samples: " + str(count / total))


def save_undersampled_data(col_count):
    undersampled_data = []
    with open('data/X_train.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if len(row) >= col_count:
                undersampled_data.append(row[:col_count])
    undersampled_data = np.asarray(undersampled_data)
    pd.DataFrame(undersampled_data).to_csv("data/X_train_undersampled.csv", index=False, header=False)

