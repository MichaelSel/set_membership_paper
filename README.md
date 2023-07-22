#Scale Studies
1. Study I - Pentatonic vs. Chromatic
2. Study II - All 66 5-note sets

## To re-plot data from the paper: (nothing needs to be done prior)
Simply go to the notebook [3. Main Plotting.ipynb](3.%20Main%20Plotting.ipynb) in the root folder and run it. 

## BUT, if you want to re-run the entire data processing from the start: 
1. Run [0. Preprocess Raw Data.ipynb](0.%20Preprocess%20Raw%20Data.ipynb) in the root folder and run it to download the raw data and perform pre-processing of the data.
2. Run [1. Data Exclusion.ipynb](1.%20Data%20Exclusion.ipynb) in the root folder and run it to see and re-apply the data exclusion criteria.

## To view some of the studies' meta-data (e.g., number of participants, button-press distributions, etc.):
For **Study 1** meta-data: Run [2a. Meta-Data Study I.ipynb](2a.%20Meta-Data%20Study%20I.ipynb) in the root folder.

For **Study 2** meta-data: Run [2b. Meta-Data Study II.ipynb](2b.%20Meta-Data%20Study%20II.ipynb) in the root folder.


Folder Structure:
<pre>
|__Shared_Scripts
    |__Download_Data.py
    |__general_funcs.py
    |__Group_Level_Set.py
    |__Plotting.ipnyb
    |__Preprocess_Qualtrics_Data.py
    |__Preprocess_Raw.py
|__Study Folder (there's a folder for each of the 2 studies)
    |__paths.py
    |__Data
      |__Experimental
        |__Processed
        |__Raw
      |__Qualtrics
        |__Processed
        |__Raw
    |__Scripts
        |__0_Preprocessing
            |__1_Preprocess_Qualtrics_Data.py
            |__2_Download_Data.py
            |__3_Preprocess_Raw.py
        |__1_Analysis
            |__Group_Level_Set
</pre>
