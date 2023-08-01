"""#########################################################################
This script:
1) Calculates totals, rates, and averages of relevant data
2) Does basic restructuring of data
3) Combines data with qualtrics data
#########################################################################"""
import numpy as np
from Study_Likert.paths import *

import pandas as pd
from scipy.stats import binomtest


def calc_group_level(processed_dir, processed_data_pickle_filename, qualtrics_processed_path):
    AT = pd.read_pickle(processed_dir + processed_data_pickle_filename)  # AT = All Trials

    ATND = AT  # ATND = All Trials No Decoys (in this task there's no decoys so we keep everything)

    # #Ignore trials with a response of 3
    # ATND = ATND[ATND['response_numeric']!=3]

    # Ignore malformed trials (only selects trials that have an empty 'malformed' field)
    ATND = ATND[ATND['malformed'] == ""]


    # Rename "sona_x" to "sona"
    ATND = ATND.rename(columns={'sona_x': 'sona'})


    # Starts a new dataframe called "GL" which will store the group-level calculated fields. The following line
    # keeps the median of the response
    GL = ATND.groupby(['sona','subject', 'set', 'stimulus']).median().reset_index()

    GL = GL.iloc[:, :3]  # Selects only the first 3 columns


    # Take median response
    median = ATND.groupby(['sona','subject', 'set', 'stimulus'])['response_numeric'].median().reset_index()
    median = median.rename(columns={'response_numeric': 'likert_median'})
    median = median.pivot(index=['sona','subject', 'set'], columns="stimulus", values="likert_median").reset_index()

    median['Q1'] = 'Some notes felt more important than others.'
    median['Q2'] = 'The audio clip was melodic.'
    median['Q3'] = 'The melody as a whole or parts of it felt familiar.'

    median = median.rename(columns={
        'Some notes felt more important than others.': 'Q1 Median',
        'The audio clip was melodic.': 'Q2 Median',
        'The melody as a whole or parts of it felt familiar.': 'Q3 Median'
    })

    # Take mean response
    mean = ATND.groupby(['sona', 'subject', 'set', 'stimulus'])['response_numeric'].mean().reset_index()
    mean = mean.rename(columns={'response_numeric': 'likert_mean'})
    mean = mean.pivot(index=['sona','subject', 'set'], columns="stimulus", values="likert_mean").reset_index()

    median['Q1'] = 'Some notes felt more important than others.'
    median['Q2'] = 'The audio clip was melodic.'
    median['Q3'] = 'The melody as a whole or parts of it felt familiar.'

    mean = mean.rename(columns={
        'Some notes felt more important than others.': 'Q1 Mean',
        'The audio clip was melodic.': 'Q2 Mean',
        'The melody as a whole or parts of it felt familiar.': 'Q3 Mean'
    })


    # Number of responses
    count = ATND.groupby(['sona','subject', 'set', 'stimulus'])['response_numeric'].count().reset_index()
    count = count.rename(columns={'response_numeric': 'likert_Qs'})
    count = count.pivot(index=['sona','subject', 'set'], columns="stimulus", values="likert_Qs").reset_index()
    count = count.rename(columns={
        'Some notes felt more important than others.': 'Q1 responses',
        'The audio clip was melodic.': 'Q2 responses',
        'The melody as a whole or parts of it felt familiar.': 'Q3 responses'
    })




    # RTs of responses
    RTs = ATND.groupby(['sona','subject', 'set', 'stimulus'])['rt'].median().reset_index()
    RTs = RTs.rename(columns={'rt': 'likert_rt'})
    RTs = RTs.pivot(index=['sona','subject', 'set'], columns="stimulus", values="likert_rt").reset_index()
    RTs = RTs.rename(columns={
        'Some notes felt more important than others.': 'Q1 RT',
        'The audio clip was melodic.': 'Q2 RT',
        'The melody as a whole or parts of it felt familiar.': 'Q3 RT'
    })

    # The original df is merged with other calculated data.
    GL = pd.merge(GL, median, on=['sona','subject', 'set'])
    GL = pd.merge(GL, mean, on=['sona','subject', 'set'])
    GL = pd.merge(GL, count, on=['sona','subject', 'set'])
    GL = pd.merge(GL, RTs, on=['sona','subject', 'set'])


    # Mark whether the subjects understood the task or not
    understood = ATND.groupby('sona')['understood task'].first().reset_index()
    GL = pd.merge(GL, understood, on=['sona'])

    GL = GL.groupby(['sona','subject','set']).first().reset_index()

    GL.to_csv(processed_dir + 'group_level_results.csv')  # Saving to file.


    # Only trials where subject understood task
    set_level = GL[GL['understood task']==True].groupby('set').median().reset_index()

    set_level = set_level[['set','Q1 Median','Q2 Median','Q3 Median', 'Q1 Mean', 'Q2 Mean','Q3 Mean', 'Q1 RT','Q2 RT','Q3 RT']]
    set_level['Q1 subjects'] = GL.groupby('set').count().reset_index()['Q1 responses']
    set_level['Q2 subjects'] = GL.groupby('set').count().reset_index()['Q2 responses']
    set_level['Q3 subjects'] = GL.groupby('set').count().reset_index()['Q3 responses']
    set_level['Q1'] = GL.groupby('set').first().reset_index()['Q1']
    set_level['Q2'] = GL.groupby('set').first().reset_index()['Q2']
    set_level['Q3'] = GL.groupby('set').first().reset_index()['Q3']


    set_level.to_csv(processed_dir + 'set_level_results.csv')  # Saving to file.

calc_group_level(processed_dir, processed_data_pickle_filename, qualtrics_processed_path)
