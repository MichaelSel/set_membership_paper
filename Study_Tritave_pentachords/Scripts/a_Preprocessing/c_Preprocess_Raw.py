import re

from Study_Tritave_pentachords.paths import *
from Shared_Scripts.Preprocess_Raw import preprocess_raw
from Shared_Scripts.general_funcs import *
from Shared_Scripts.key_finder import get_key_r



def reformat_data(subject_pattern, data_dir='./raw_data', processed_dir='./processed', trial_name='choice'):
    """This function takes the stimuli jsons and the response jsons for each subject and combines them into one big
    dataframe with all the subjects and all the trials"""

    # this will hold all of the trials for *all* of the subjects
    data = []

    # gets all the folders who fit the prefix of the task sets. Each folder is the data of a single subject.
    subjects = [{'id': x, 'dir': data_dir + "/" + x + "/csv"} for x in os.listdir(data_dir) if (bool(re.match(subject_pattern, x)))]

    # iterates through all of the subjects:
    for subject in subjects:
        # grabs the paths of the stimuli files for all of the blocks
        stimuli_files = [f for f in os.listdir(subject['dir']) if os.path.isfile(os.path.join(subject['dir'], f)) and (
                f.startswith("block") and f.endswith(".json"))]

        # grabs the paths of the response files for all of the blocks
        resp_files = [subject['id'] + "_" + f for f in stimuli_files]

        if len(stimuli_files) == 0:
            # If the subject folder is empty, move to the next subject
            continue

        if not len(resp_files) == len(stimuli_files):
            # if subject did not complete the task (fewer response files than stimuli files), skip to next subject.
            continue

        # Iterate block by block
        for block_num in range(len(stimuli_files)):
            block_data = []
            # load the json for the stimuli
            stim_block_json = get_json(subject['dir'] + "/" + stimuli_files[block_num])
            try:
                # attempt to grab the json for the responses
                resp_block_json = get_json(subject['dir'] + "/" + resp_files[block_num])
            except:
                # if there was an error, move on to the next iteration.
                continue
            # From the response files only keep the entries holding the relevant trial types (we don't need data about
            # fixation crosses or instructions displayed)
            resp_block_json = [entry for entry in resp_block_json if entry['name'] == trial_name]
            # Iterate trial by trial
            for trial_num in range(len(stim_block_json)):
                # combine the contents of the trial data from the stimuli file as well as the trial data from the
                # response file.
                trial = {**stim_block_json[trial_num], **resp_block_json[trial_num]}  # merging the files
                trial['set']=trial['type']
                # add a field called "chose" who explicitly states what the subject chose (shifted/swapped/neither)
                if trial['response'] == '1st':
                    trial['chose'] = trial['order'][0]
                elif trial['response'] == '2nd':
                    trial['chose'] = trial['order'][1]
                elif trial['response'] == 'neither':
                    trial['chose'] = "neither"
                else:  # If somehow some other value appeared, show an error. This shouldn't happen.
                    print("error")

                # Add a field that specifies the length of the melody in that trial
                trial['length'] = len(trial['probe'])
                block_data.append(trial)  # append the data
            # print(subject['dir'] + "/" + resp_files[block_num])

            json_export = json.dumps(block_data)
            full_path = subject['dir'] + "/" + resp_files[block_num]
            f = open(full_path, "w")
            f.write(json_export)
            f.close()
            # print("Re-saving {}.".format(subject['id']))
reformat_data(SUBJECT_PATTERN, raw_data_dir, processed_dir)

preprocess_raw(SUBJECT_PATTERN, raw_data_dir, qualtrics_processed_path, processed_dir, processed_data_pickle_filename,
               processed_data_csv_filename)


print("Preprocessing complete.")