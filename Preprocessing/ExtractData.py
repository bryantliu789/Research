import csv
import glob
import os
import pickle
import re
from pprint import pprint
import sys

import numpy as np
import pandas as pd

pickle_file = "data.pkl"

required_cols = [26, 28, 30]

rows_to_skip = 4000
rows_to_read = 6000

data = {}
labels = ['event_type', 'C3', 'Cz', 'C4']
for label in labels:
    data[label] = list()

subfolders = [f.path for f in os.scandir('.') if f.is_dir()]

for subfolder in list(subfolders):
    if not re.search("^./patient\d+session\d+_\d+", subfolder):
        continue
    print("\nScanning folder:", subfolder)

    evtype_csv = glob.glob(subfolder + "/eventtype*.csv")
    if len(evtype_csv) != 1:
        raise RuntimeError("No unique eventtype CSV file in " + subfolder + "?")
    print("  Found event-type file:", evtype_csv[0])

    ev_types = []
    with open(evtype_csv[0], 'r') as f_ev:
        rows = csv.reader(f_ev)
        ev_types = list(map(int, next(rows)))

    index = 0

    files = glob.glob(subfolder + "/*patient*session*trial*target*result*.csv")
    for file in files:
        print("  Reading file:", file)

        columns = []
        with open(file, 'r') as f_trial:
            rows = csv.reader(f_trial)
            try:
                for i in range(rows_to_skip):
                    row = next(rows)
            except Exception as e:
                print(f"Exception {e} reading {file}")
                break

            try:
                for i in range(rows_to_read):
                    row = next(rows)
                    columns.append([float(row[j - 1]) for j in required_cols])
            except Exception as e:
                print(f"Exception {e} reading {file}")
                break

        columns = list(map(list, zip(*columns)))

        n_entries = len(columns[:][0])
        if n_entries < rows_to_read:
            print(f"Not enough entries ({n_entries} < {rows_to_read}), skipping this file")
            continue

        data['event_type'].append(ev_types[index])
        data['C3'].append(np.array(columns[:][0]))
        data['Cz'].append(np.array(columns[:][1]))
        data['C4'].append(np.array(columns[:][2]))
        index += 1

frame = pd.DataFrame(data=data, columns=labels)
pickle_out = open(pickle_file, "wb")
pickle.dump(frame, pickle_out)
pickle_out.close()