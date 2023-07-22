"""#########################################################################
This script:
1) Combines all the responses by all the subjects into 1 big file
2) Does basic restructuring of data
3) Combines data with qualtrics data
#########################################################################"""
import os
import json
import pandas as pd
from datetime import datetime
import re


def preprocess_raw(subject_pattern, raw_data_dir, qualtrics_processed_path, processed_dir,
                   processed_data_pickle_filename, processed_data_csv_filename):
    all_responses = []  # Will hold all the responses in one frame
    r = re.compile(subject_pattern)
    dirs = os.listdir(raw_data_dir)
    dirs = [dir for dir in dirs if r.match(dir)]
    for directory in dirs:  # Iterating through each subject directory
        print("Processing:", directory)
        for filename in os.listdir(raw_data_dir + directory + "/csv"):  # looks in the /csv directory of each subject
            # if filename matches subject formatting
            if bool(re.match(subject_pattern, filename)):
                with open(raw_data_dir + directory + "/csv/" + filename) as json_file:
                    data = json.load(json_file)  # Loads the data as JSON.
                    for response in data:  # each line in the JSON is a trial

                        response['malformed'] = []

                        # Other data is stored in the JSON, but we only care about the "choice" data which is the choice
                        # the subject made upon hearing the melodies
                        if (response['name'] == 'choice'):
                            # By a previous coding error, a few responses did not store the set that was used to generate
                            # them. Therefore, those are skipped
                            if ('set' not in response):
                                response['malformed'].append("No 'set' field")
                                continue

                            # The transposition of the trial
                            transposition = response['transposition']

                            # Extracts the set from the probe
                            probe = response['probe_pitches']
                            probe_from0 = [(pitch - transposition) % 12 for pitch in probe]
                            probe_set = list(set(probe_from0))

                            # Extracts the set from the shifted version
                            shifted = response['shifted_pitches']
                            shifted_from0 = [(pitch - transposition) % 12 for pitch in shifted]
                            shifted_set = list(set(shifted_from0))

                            # Extracts the set from the swapped version
                            swapped = response['swapped_pitches']
                            swapped_from0 = [(pitch - transposition) % 12 for pitch in swapped]
                            swapped_set = list(set(swapped_from0))

                            # Excludes trials that don't use all 5 of the set's notes.
                            if (len(probe_set) < 5):
                                response['malformed'].append("probe doesn't include all set's notes.")

                            # Sanity: make sure that the probe and swapped version have the same set, and that the probe
                            # and shifted version do not.
                            if ((probe_set != swapped_set) or (probe_set == shifted_set)):
                                response['malformed'].append("probe, shifted, and swapped disagree.")

                            # Re-format lists as [space] delineated strings ([0,1,2] into "0 1 2" for sets)
                            # Addressing the set as a string rather than as a list makes things easier down the line.
                            # If the set is saved as string already (e.g., "best", "chromatic", etc.), it does nothing
                            if isinstance(response['set'], list):
                                response['set'] = [str(int) for int in response['set']]
                                response['set'] = ' '.join(response['set'])

                            # The same reasoning applied to the probe's pitches
                            response['probe_pitches'] = [str(int) for int in response['probe_pitches']]
                            response['probe_pitches'] = ' '.join(response['probe_pitches'])

                            # Same for the shifted melody's pitches
                            response['shifted_pitches'] = [str(int) for int in response['shifted_pitches']]
                            response['shifted_pitches'] = ' '.join(response['shifted_pitches'])

                            # Same for the swapped melody's pitches
                            response['swapped_pitches'] = [str(int) for int in response['swapped_pitches']]
                            response['swapped_pitches'] = ' '.join(response['swapped_pitches'])

                            # Store the "order" list seperately as "option 1" and "option 2" the order in which the
                            # subject heard the test melodies
                            response['option_1'] = response['order'][0]
                            response['option_2'] = response['order'][1]

                            # Convert "time" from a string to a datetime object Note: It appears that the time entry was
                            # not stored correctly (within a subject the time appears to not change), and to calculate
                            # RTs we'll use the designated RT field or the time_elapsed field.
                            response['time'] = datetime.fromisoformat(response['time'].replace("Z", "+00:00"))

                            # Sanity check: Verify the "char" field (button pressed) and the "chose" fields agree.
                            if (response['response'] == "1st" and response['char'] != "A") or (
                                    response['response'] == "2nd" and response['char'] != "L") or (
                                    response['response'] == "neither" and response['char'] != "G"):
                                response['malformed'].append("Key presses don't agree")

                            # Store the subject's response contextually: shifted, swapped, or neither (rather than 1st, 2nd, or neither)
                            if response['response'] == "1st":
                                response['chose'] = response['option_1']
                            elif response['response'] == "2nd":
                                response['chose'] = response['option_2']
                            elif response['response'] == "neither":
                                response['chose'] = "neither"
                            # There shouldn't be any other options, so if there is somehow an additional one,
                            # it requires some investigation.
                            else:
                                response['chose'] = "Error"
                                response['malformed'].append("Unavailable option.")

                            response['malformed'] = "; ".join(response['malformed'])

                            # Append each response to the list of all responses for all subjects
                            all_responses.append(response)

    # Once all responses have been appended, reformat the list of responses into a pd Dataframe.
    all_responses = pd.DataFrame.from_dict(all_responses)

    if 'subject' not in all_responses.columns:
        all_responses = all_responses.rename(columns={'sub': 'subject'})

    # To ensure the order of the dataframe we sort the values first by subject (so a subject's responses are appearing in
    # a row), Then within that subject we sort by time, and then by time elapsed (the 'time' parameter can probably be
    # omitted).
    all_responses = all_responses.sort_values(by=['subject', 'time', 'time_elapsed']).reset_index()

    # Combine with qualtrics based on subject ID
    qualtrics = pd.read_csv(qualtrics_processed_path)
    # on="subject" means that the 'subject' on either dataframe corresponded to the same thing and therefore that column
    # should be the basis for merging.
    all_responses = pd.merge(all_responses, qualtrics, on="subject")

    # Flag subjects who didn't understand the task
    all_responses['understood task'] = all_responses['I understood the instructions of the task'].isin(
        ['Strongly Agree', 'Agree'])

    # Saving the dataframe to a pickle (better than CSV because it remembers variable object types)
    all_responses.to_pickle(processed_dir + processed_data_pickle_filename)
    all_responses.to_csv(processed_dir + processed_data_csv_filename)

    print("Data saved to {}".format(processed_dir + processed_data_pickle_filename))
