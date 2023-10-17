import math

import numpy as np
import pandas as pd

from Study_Uniform.paths import *
from Shared_Scripts.Preprocess_Raw import preprocess_raw
from Shared_Scripts.key_finder import get_key_r

preprocess_raw(SUBJECT_PATTERN, raw_data_dir, qualtrics_processed_path, processed_dir, processed_data_pickle_filename,
               processed_data_csv_filename)

def countInterval(entry, size):
    entry_as_list = [int(el) for el in entry.split(" ")]
    entry_deltas = [abs(j - i) for i, j in zip(entry_as_list[:-1], entry_as_list[1:])]
    return entry_deltas.count(size)


# Load data
all_responses = pd.read_pickle(processed_dir + processed_data_pickle_filename)

print("Counting I7")
all_responses['I7_count'] = all_responses.apply(lambda row: countInterval(row['probe_pitches'], 7), axis=1)
print("Counting I5")
all_responses['I5_count'] = all_responses.apply(lambda row: countInterval(row['probe_pitches'], 5), axis=1)
print("Counting I1")
all_responses['I1_count'] = all_responses.apply(lambda row: countInterval(row['probe_pitches'], 1), axis=1)
print("Counting I11")
all_responses['I11_count'] = all_responses.apply(lambda row: countInterval(row['probe_pitches'], 11), axis=1)


print("Adding set features")
# Combine with set features based on set
features = pd.read_csv(ROOT_DIR + "/5-note-sets-with-features.csv")
all_responses = pd.merge(all_responses, features, on="set")
all_responses = all_responses.sort_values(by="RecordedDate").reset_index(drop=True)


print("Adding key finding r. NOTE: This takes a while. (~30 minutes.)")
# add key finding stuff
all_responses['key_r'] = all_responses.apply(
    lambda row: get_key_r([int(n) % 12 for n in row['probe_pitches'].split()])[0], axis=1)

# Creating 6 sections
total_responses = all_responses.shape[0]
sixth = math.ceil(total_responses / 6)
all_responses['section'] = all_responses.index / sixth
all_responses['section'] = all_responses['section'].apply(np.floor)

all_responses.to_pickle(processed_dir + processed_data_pickle_filename)
all_responses.head(1000).to_csv(processed_dir + processed_data_csv_filename)

print("Preprocessing complete.")
