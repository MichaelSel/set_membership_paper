import os

SUBJECT_PATTERN = 'ATON'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is where this file is
DATA_DIR = ROOT_DIR + "/Data/"
qualtrics_dir = ROOT_DIR + '/Data/Qualtrics/Raw/'
qualtrics_processed_path = ROOT_DIR + '/Data/Qualtrics/Processed/qualtrics.csv'
raw_data_dir = ROOT_DIR + '/Data/Experimental/Raw/'
processed_dir = ROOT_DIR + '/Data/Experimental/Processed/'
processed_data_pickle_filename = "single_trial_results.pickle"
processed_data_csv_filename = "single_trial_results-DEMO.csv"
plots_dir = ROOT_DIR + '/Plots/'
post_exclusion_data_pickle_filename = "PE_group_level.pickle"

