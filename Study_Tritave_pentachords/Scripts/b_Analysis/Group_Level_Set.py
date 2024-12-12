"""#########################################################################
This script:
1) Calculates totals, rates, and averages of relevant data
2) Does basic restructuring of data
3) Combines data with qualtrics data
#########################################################################"""
import numpy as np
import pandas as pd
from Study_Tritave_pentachords.paths import *
from Shared_Scripts.Group_Level_Set import calc_group_level

calc_group_level(processed_dir,processed_data_pickle_filename,qualtrics_processed_path, run_additional_computations=False)

# Add additional stuff
GL = pd.read_csv(processed_dir + 'group_level_results.csv')  # Opening the file

# Rename values in column "set" from 237 to "0 2 3 7"
GL['set'] = GL['set'].apply(lambda x: '0 1 2 4 8' if x == 1248 else x)
GL['set'] = GL['set'].apply(lambda x: '0 2 4 7 9' if x == 2479 else x)


GL.to_csv(processed_dir + 'group_level_results.csv')  # Saving to file.
