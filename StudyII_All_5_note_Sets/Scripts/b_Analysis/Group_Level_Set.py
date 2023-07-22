"""#########################################################################
This script:
1) Calculates totals, rates, and averages of relevant data
2) Does basic restructuring of data
3) Combines data with qualtrics data
#########################################################################"""
import numpy as np
import pandas as pd
from StudyII_All_5_note_Sets.paths import *
from Shared_Scripts.Group_Level_Set import calc_group_level

calc_group_level(processed_dir,processed_data_pickle_filename,qualtrics_processed_path, run_additional_computations=True)
calc_group_level(processed_dir,processed_data_pickle_filename,qualtrics_processed_path,has_decoy=True,filename='decoys.csv', run_additional_computations=True)

# Add additional stuff
GL = pd.read_csv(processed_dir + 'group_level_results.csv')  # Opening the file
GL_decoys = pd.read_csv(processed_dir + 'decoys.csv')  # Opening the file




AT = pd.read_pickle(processed_dir + processed_data_pickle_filename)  # AT = All Trials
ATND = AT[AT['has_decoy'] == False]  # ATND = All Trials No Decoys

# Ignore malformed trials (only selects trials that have an empty 'malformed' field)
# ATND = ATND[ATND['malformed'] == ""]

# Add the section number of each subject
temp = ATND.groupby(['subject'])['section'].mean().reset_index()
GL = pd.merge(GL, temp, on=['subject'])
GL['section'] = GL['section'].apply(np.floor)

# Combine with set features based on set
features = pd.read_csv(ROOT_DIR + "/5-note-sets-with-features.csv")
GL = pd.merge(GL, features, on="set")
GL_decoys = pd.merge(GL_decoys, features, on="set")

GL.to_csv(processed_dir + 'group_level_results.csv')  # Saving to file.
GL_decoys.to_csv(processed_dir + 'group_level_decoy_results.csv')