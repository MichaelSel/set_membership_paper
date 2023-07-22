"""#########################################################################
This script:
1) Calculates totals, rates, and averages of relevant data
2) Does basic restructuring of data
3) Combines data with qualtrics data
#########################################################################"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

def runs_test(sequence,element1="A",element2="B",passing_z = 1.645):
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm
    from itertools import groupby
    from collections import Counter
    import math

    counts = Counter(k for k, g in groupby(sequence))

    n1 = sequence.count(element1) # number of tails
    n2 = sequence.count(element2) # number of heads
    if n1==0 or n2==0: return 0
    R = len([(k, list(g)) for k, g in groupby(sequence)]) # how many times the sequence alternates

    R_ = ((n1*n2*2)/(n1+n2))+1 # the expected number of runs

    s2r = (2*n1*n2 * (2*n1*n2-n1-n2))/(pow(n1+n2,2)*(n1+n2-1))
    sr = math.sqrt(s2r)
    if(sr==0): Z=0
    else:
        Z = (R-R_)/sr
    passing = abs(Z)>passing_z
    return Z
    ## The following data is relevant when both n1 and n2 are greater than 10
    # p	     0.001	 0.005	 0.010	 0.025	 0.050	 0.100
    # Zp    -3.090  -2.576  -2.326  -1.960  -1.645  -1.282

    # p   0.999    0.995    0.990    0.975    0.950    0.900
    # Zp +3.090   +2.576   +2.326   +1.960   +1.645   +1.282

def get_binom_p(k,n):
    if(n>0):
        result = binomtest(k,n,.5,'greater')
        p = result.pvalue
        return p
    else:
        return None


def calc_group_level(processed_dir, processed_data_pickle_filename, qualtrics_processed_path, has_decoy=False, filename="group_level_results.csv", run_additional_computations=False):
    AT = pd.read_pickle(processed_dir + processed_data_pickle_filename)  # AT = All Trials

    AT = AT[(AT['has_decoy'] == has_decoy)]


    # Ignore malformed trials (only selects trials that have an empty 'malformed' field)
    AT = AT[AT['malformed'] == ""]

    # Starts a new dataframe called "GL" which will store the group-level calculated fields. The following line
    # counts how many times a certain choice (e.g. 'shifted') was made within a certain set  (e.g., the pentatonic),
    # within a certain subject (e.g., SSS0001)
    GL = AT.groupby(['subject', 'set']).count().reset_index()

    GL = GL.iloc[:, :2]  # Selects only the first 3 columns

    # Mark whether the subjects understood the task or not
    understood = AT.groupby('subject')['understood task'].first().reset_index()
    GL = pd.merge(GL, understood, on=['subject'])

    # Mark whether the subject passes a runs test on their button distribution (chances they don't have a button pattern)



    choices = AT[AT['response'] != "neither"]
    choices['response'] = choices['response'].str.replace('1st','A')
    choices['response'] = choices['response'].str.replace('2nd','B')
    choices = choices.groupby(['subject']).agg({'response': ''.join}).reset_index()
    choices['button_bias_Z'] = choices['response'].apply(runs_test)

    GL = pd.merge(GL, choices, on=['subject'], how="left")

    if(has_decoy):
        decoys = AT[AT['has_decoy'] == True] ## only responses that have decoys
        decoys_NN = decoys[decoys['response'] != "neither"]
        decoy_NN_count = decoys_NN.groupby(['subject']).count()['chose'].reset_index().rename(columns={'chose':'decoy_NN_count'})
        decoy_NN_count_correct = decoys_NN[decoys_NN['chose'] == "shifted"].groupby(['subject']).count()['chose'].reset_index().rename(columns={'chose':'decoy_NN_count_correct'})

        GL = pd.merge(GL, decoy_NN_count, on=['subject'], how="left")
        GL = pd.merge(GL, decoy_NN_count_correct, on=['subject'], how="left")
        GL['decoy_NN_count'] = GL['decoy_NN_count'].fillna(0)
        GL['decoy_NN_count_correct'] = GL['decoy_NN_count_correct'].fillna(0)

        GL['decoy_NN_count'] = GL['decoy_NN_count'].astype(int)
        GL['decoy_NN_count_correct'] = GL['decoy_NN_count_correct'].astype(int)

        GL['decoy_binom'] = GL.apply(lambda x: get_binom_p(x['decoy_NN_count_correct'],x['decoy_NN_count']), axis=1)


    # We also calculate the mean RT within the same grouping (of a certain choice, within a certain set, within a certain
    # subject)
    set_rt = AT.groupby(['subject', 'set'])['rt'].median().reset_index()
    set_rt = set_rt.rename(columns={'rt': 'rt set'})

    shifted_rt = AT[AT['chose'] == 'shifted'].groupby(['subject', 'set'])['rt'].median().reset_index()
    shifted_rt = shifted_rt.rename(columns={'rt': 'rt shifted'})

    swapped_rt = AT[AT['chose'] == 'swapped'].groupby(['subject', 'set'])['rt'].median().reset_index()
    swapped_rt = swapped_rt.rename(columns={'rt': 'rt swapped'})

    neither_rt = AT[AT['chose'] == 'neither'].groupby(['subject', 'set'])['rt'].median().reset_index()
    neither_rt = neither_rt.rename(columns={'rt': 'rt neither'})

    # The df is merged with the median RTs.
    GL = pd.merge(GL, set_rt, on=['subject', 'set'], how="left")
    GL = pd.merge(GL, shifted_rt, on=['subject', 'set'], how="left")
    GL = pd.merge(GL, swapped_rt, on=['subject', 'set'], how="left")
    GL = pd.merge(GL, neither_rt, on=['subject', 'set'], how="left")

    if(run_additional_computations):
        #key-finding r avg:
        key_r = AT.groupby(['subject', 'set'])['key_r'].mean().reset_index()
        GL = pd.merge(GL, key_r, on=['subject', 'set'], how="left")

        # Actual interval appearance
        I7 = AT.groupby(['subject', 'set'])['I7_count'].mean().reset_index()
        I5 = AT.groupby(['subject', 'set'])['I5_count'].mean().reset_index()
        I1 = AT.groupby(['subject', 'set'])['I1_count'].mean().reset_index()
        I11 = AT.groupby(['subject', 'set'])['I11_count'].mean().reset_index()

        GL = pd.merge(GL, I7, on=['subject', 'set'], how="left")
        GL = pd.merge(GL, I5, on=['subject', 'set'], how="left")
        GL = pd.merge(GL, I1, on=['subject', 'set'], how="left")
        GL = pd.merge(GL, I11, on=['subject', 'set'], how="left")


    GL['rt shifted-swapped'] = GL['rt shifted'] - GL['rt swapped']
    GL['rt shifted:swapped'] = GL['rt shifted'] / GL['rt swapped']

    if ('length' in AT.columns):
        temp = AT.groupby('subject')['length'].mean().reset_index()
        GL = pd.merge(GL, temp, on=['subject'])



    # Count trials for each condition
    shifted_count = AT[AT['chose'] == 'shifted'].groupby(['subject', 'set'])['index'].count().reset_index()
    shifted_count = shifted_count.rename(columns={'index': '# shifted'})

    swapped_count = AT[AT['chose'] == 'swapped'].groupby(['subject', 'set'])['index'].count().reset_index()
    swapped_count = swapped_count.rename(columns={'index': '# swapped'})

    neither_count = AT[AT['chose'] == 'neither'].groupby(['subject', 'set'])['index'].count().reset_index()
    neither_count = neither_count.rename(columns={'index': '# neither'})

    # The df is merged with the mean number of trials for each condition.
    GL = pd.merge(GL, shifted_count, on=['subject', 'set'], how="left")
    GL = pd.merge(GL, swapped_count, on=['subject', 'set'], how="left")
    GL = pd.merge(GL, neither_count, on=['subject', 'set'], how="left")

    # Count trials for each button for each subject for each set
    first_count = AT[AT['response'] == '1st'].groupby(['subject', 'set'])['index'].count().reset_index()
    first_count = first_count.rename(columns={'index': '# 1st button'})

    second_count = AT[AT['response'] == '2nd'].groupby(['subject', 'set'])['index'].count().reset_index()
    second_count = second_count.rename(columns={'index': '# 2nd button'})

    neitherB_count = AT[AT['response'] == 'neither'].groupby(['subject', 'set'])['index'].count().reset_index()
    neitherB_count = neitherB_count.rename(columns={'index': '# neither button'})

    # The df is merged with the mean number of trials for each condition.
    GL = pd.merge(GL, first_count, on=['subject', 'set'], how="left")
    GL = pd.merge(GL, second_count, on=['subject', 'set'], how="left")
    GL = pd.merge(GL, neitherB_count, on=['subject', 'set'], how="left")

    # Count trials for each button for each subject across all sets
    first_count = AT[AT['response'] == '1st'].groupby(['subject'])['index'].count().reset_index()
    first_count = first_count.rename(columns={'index': '# 1st button (task)'})

    second_count = AT[AT['response'] == '2nd'].groupby(['subject'])['index'].count().reset_index()
    second_count = second_count.rename(columns={'index': '# 2nd button (task)'})

    neitherB_count = AT[AT['response'] == 'neither'].groupby(['subject'])['index'].count().reset_index()
    neitherB_count = neitherB_count.rename(columns={'index': '# neither button (task)'})

    # The df is merged with the mean number of trials for each condition.
    GL = pd.merge(GL, first_count, on=['subject'], how="left")
    GL = pd.merge(GL, second_count, on=['subject'], how="left")
    GL = pd.merge(GL, neitherB_count, on=['subject'], how="left")

    # Replace NaN with 0s
    GL =  GL.fillna(0)

    # holds the TOTAL number of button presses for that subject across entire task.
    GL['# button presses (task)'] = GL['# 1st button (task)'] + GL['# 2nd button (task)'] + GL[
        '# neither button (task)']

    # holds the TOTAL number of trials that subject saw for that set.
    GL['# trials'] = GL['# shifted'] + GL['# swapped'] + GL['# neither']

    # holds the TOTAL number of button presses for that set.
    GL['# button presses'] = GL['# 1st button'] + GL['# 2nd button'] + GL['# neither button']

    # The total number of trials if we ignore neithers.
    GL['# no_neither_trials'] = GL['# trials'] - GL['# neither']

    # Iterates through the different conditions (shifted, swapped, neither) and calculates their rate (0 through 1).
    GL['rate shifted'] = GL['# shifted'] / GL['# trials']
    GL['rate swapped'] = GL['# swapped'] / GL['# trials']
    GL['rate neither'] = GL['# neither'] / GL['# trials']

    # Iterates through the different button presses (1st, 2nd, neither) and calculates their rate (0 through 1).
    GL['rate pressed 1st'] = GL['# 1st button'] / GL['# button presses']
    GL['rate pressed 2nd'] = GL['# 2nd button'] / GL['# button presses']
    GL['rate pressed neither'] = GL['# neither button'] / GL['# button presses']

    # Iterates through the different button presses (1st, 2nd, neither) and calculates their rate (0 through 1) across task.
    GL['rate pressed 1st (task)'] = GL['# 1st button (task)'] / GL['# button presses (task)']
    GL['rate pressed 2nd (task)'] = GL['# 2nd button (task)'] / GL['# button presses (task)']
    GL['rate pressed neither (task)'] = GL['# neither button (task)'] / GL['# button presses (task)']

    # Calculates the rate of shifted and swapped when neithers are ignored. (NN=No Neithers)
    GL['rate_NN_shifted'] = GL['# shifted'] / GL['# no_neither_trials']
    GL['rate_NN_swapped'] = GL['# swapped'] / GL['# no_neither_trials']

    GL['rate shifted - rate swapped'] = GL['rate shifted'] - GL['rate swapped']
    GL['rate shifted - rate swapped (NN)'] = GL['rate_NN_shifted'] - GL['rate_NN_swapped']

    # Merge with qualtrics so richer crossections can be achieved. (The same code appear in 3_reprocess_raw.py)
    qualtrics = pd.read_csv(qualtrics_processed_path)
    GL = pd.merge(GL, qualtrics, on="subject")

    GL.to_csv(processed_dir + filename)  # Saving to file.
