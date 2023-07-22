"""#########################################################################
This script:
Reads the processed qualtrics CSV, and based on its entries downloads the relevant raw data into the raw data folder

Note: for this to work, the AWS CLI must be installed, otherwise the downloading must be done manually.
Also Note: This may take a long time. (hours)
#########################################################################"""

import os
import pandas as pd

def download_data(qualtrics_path,raw_data_dir):
    # Loads the processed qualtrics csv we created in 1_Preprocess_Qualtrics_data.py
    qualtrics = pd.read_csv(qualtrics_path)  # path of qualtrics survey with all relevant responses

    # extract all subjects (task set IDs) from qualtrics data
    subjects = qualtrics['subject'].tolist()[2:]

    # Get folders already in folder:
    dirs = [dir for dir in os.listdir(raw_data_dir)]

    # Download each subject's task set
    for SubjectID in subjects:
        if(SubjectID in dirs):
            print("Subject {} already exists, moving to the next subject.".format(SubjectID))
            continue
        print(raw_data_dir)
        print("Downloading data for subject: ", SubjectID)
        os.system('aws s3 cp s3://cadseg/{0} {1}{0} --exclude="*audio*" --exclude="*images*" --recursive'
                  .format(SubjectID,raw_data_dir))  # Runs using the AWS CLI
