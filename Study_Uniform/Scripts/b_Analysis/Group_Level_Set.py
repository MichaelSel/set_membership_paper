"""#########################################################################
This script:
1) Calculates totals, rates, and averages of relevant data
2) Does basic restructuring of data
3) Combines data with qualtrics data
#########################################################################"""
import numpy as np
import pandas as pd
from Study_Uniform.paths import *
from Shared_Scripts.Group_Level_Set import calc_group_level

calc_group_level(processed_dir,processed_data_pickle_filename,qualtrics_processed_path, run_additional_computations=True)

# Add additional stuff
GL = pd.read_csv(processed_dir + 'group_level_results.csv')  # Opening the file



AT = pd.read_pickle(processed_dir + processed_data_pickle_filename)  # AT = All Trials
ATND = AT[AT['has_decoy'] == False]  # ATND = All Trials No Decoys

# Ignore malformed trials (only selects trials that have an empty 'malformed' field)
# ATND = ATND[ATND['malformed'] == ""]

# Add the section number of each subject
if('section' in GL.columns):
    temp = ATND.groupby(['subject'])['section'].mean().reset_index()
    GL = pd.merge(GL, temp, on=['subject'])
    GL['section'] = GL['section'].apply(np.floor)

# Combine with set features based on set
features = pd.read_csv(ROOT_DIR + "/uniform-sets-with-features.csv")
#if set is '0 3 6 9' rename it to 'diminished', and if it's '0 2 4 6 8 10' rename it to 'wholetone'
features['set'] = features['set'].apply(lambda x: 'diminished' if x == '0 3 6 9' else x)
features['set'] = features['set'].apply(lambda x: 'wholetone' if x == '0 2 4 6 8 10' else x)

GL = pd.merge(GL, features, on="set")

GL.to_csv(processed_dir + 'group_level_results.csv')  # Saving to file.
