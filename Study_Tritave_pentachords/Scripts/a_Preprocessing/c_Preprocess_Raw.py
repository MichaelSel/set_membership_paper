import math
import re

import numpy as np

from Study_Tritave_pentachords.paths import *
from Shared_Scripts.Preprocess_Raw import preprocess_raw
from Shared_Scripts.general_funcs import *
from Shared_Scripts.key_finder import get_key_r

preprocess_raw(SUBJECT_PATTERN, raw_data_dir, qualtrics_processed_path, processed_dir, processed_data_pickle_filename,
               processed_data_csv_filename)


# Load data
all_responses = pd.read_pickle(processed_dir + processed_data_pickle_filename)

all_responses = all_responses.sort_values(by="RecordedDate").reset_index(drop=True)


# Creating 6 sections
total_responses = all_responses.shape[0]
sixth = math.ceil(total_responses / 6)
all_responses['section'] = all_responses.index / sixth
all_responses['section'] = all_responses['section'].apply(np.floor)

all_responses.to_pickle(processed_dir + processed_data_pickle_filename)
all_responses.head(1000).to_csv(processed_dir + processed_data_csv_filename)

print("Preprocessing complete.")