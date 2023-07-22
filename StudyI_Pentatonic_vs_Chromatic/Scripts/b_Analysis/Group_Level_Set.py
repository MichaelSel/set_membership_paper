"""#########################################################################
This script:
1) Calculates totals, rates, and averages of relevant data
2) Does basic restructuring of data
3) Combines data with qualtrics data
#########################################################################"""
from StudyI_Pentatonic_vs_Chromatic.paths import *
from Shared_Scripts.Group_Level_Set import calc_group_level


calc_group_level(processed_dir,processed_data_pickle_filename,qualtrics_processed_path, run_additional_computations=False)
