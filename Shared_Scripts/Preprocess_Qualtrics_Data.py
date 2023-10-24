"""#########################################################################
This script:
1) merges multiple qualtrics csvs (from multiple surveys) if more than one exits
2) only keeps the relevant task sets (e.g., SSS... and SSS2v....)
3) Keeps only the most recent taskset when more than one subject completed the same taskset (e.g., if on Monday SSS0001
was completed and on Tuesday SSS0001 was completed by another participant. Only the Tuesday data will be saved.)
4) Saves the cleaned up and merged file
#########################################################################"""

import pandas as pd
import os

def preprocess_qualtrics(qualtrics_dir, subject_format, destination_file, rename_schema, drop_schema):
    data_frames = []
    # loop through survey files and store all completed set names
    for filename in os.listdir(qualtrics_dir):
        if filename.endswith(".csv"):
            # load survey data
            curr_survey = pd.read_csv(qualtrics_dir + filename)

            # drop rows 1 and 2 (they contain redundant qualtrics headers)
            curr_survey = curr_survey.iloc[2:, :]

            #append to main frame
            data_frames.append(curr_survey)

    # The qualtrics files contain all the subjects for the 66-set study (SSS and SSS2v), the likert-scale study (SSSQ) and
    # other unrelated studies (OVS and NOVS). The following lines filter only the relevant sets.
    qualtrics = pd.concat(data_frames, sort=True, ignore_index=True)

    # We only keep sets that have a prefix of SSS + number (e.g., SSS0001, but also SSS2v0001)
    # qualtrics = qualtrics.loc[lambda x: x['sub'].str.contains('SSS\d', regex=True)]  # regex for SSS+digit
    qualtrics = qualtrics.loc[lambda x: x['sub'].str.contains(subject_format, regex=True)]  # regex for SSS+digit

    # Some sets have been taken by more than one participant. We only keep the most recent one.
    # We sort by sub (task-set) and by 'EndDate', and then we only keep that first row for each 'sub'
    qualtrics = qualtrics.sort_values(by=['sub', 'EndDate'], ascending=False).groupby('sub', as_index=False).first()

    # Renames columns
    qualtrics = qualtrics.rename(columns=rename_schema)

    # Drops unwanted columns
    qualtrics = qualtrics.drop(drop_schema, axis=1,errors='ignore')


    # Save the relevant data to file
    qualtrics.to_csv(destination_file)

    print("Processed Qualtrics data saved at", destination_file)
