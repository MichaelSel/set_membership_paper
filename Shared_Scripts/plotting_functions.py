import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Shared_Scripts import stat_funcs
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np
from sklearn.linear_model import LinearRegression

#Helper function for plot annotation
from Shared_Scripts.stat_funcs import bootstrap_ci, permtest_coeffs, ridge_regression


def annotate(ax, data, x, y, type='pearson', itirations = 10000):
    # slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(x=data[x], y=data[y])
    if(type=='pearson'):
        r,p = stat_funcs.permtest_corr_pearson(data[x],data[y],itirations,False)
    elif(type=='spearman'):
        r, p = stat_funcs.permtest_corr(data[x], data[y], itirations)
    ax.text(.02, .9, f'r2={r ** 2:.2f}, p={p:.2g}', transform=ax.transAxes)

def bin_by_feature_bars (dataset,x,y="rate shifted - rate swapped (NN)", bins=3,diatonic="include",normalize=True,figsize=(6,6),ylim=None, show_data_points=True):
    temp = dataset #loading dataset
    temp = temp.groupby("set").mean().reset_index() #collapsing subject by set
    if(diatonic=="exclude"): temp = temp[temp['subset_of_diatonic']==False]
    elif(diatonic=="only"): temp = temp[temp['subset_of_diatonic']==True]

    if(normalize):
        min = temp[x].min()
        max = temp[x].max()
        temp[x] = (temp[x]-min)/(max-min)

    temp[x] = pd.qcut(temp[x],bins,precision=2,duplicates="drop") # cut the data into bins

    #plot the data
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("ticks")
    sns.set_context("talk")
    sns.barplot(x=x, y=y, data=temp, ax=ax, color=".9")
    if(show_data_points): sns.stripplot(x=x, y=y, data=temp, ax=ax, dodge=True, size=13, marker=".", edgecolor="gray", alpha=.8)
    if(ylim):plt.ylim(ylim)

def distribution(dataset, x,collapse_sets=False):
    temp = dataset #loading dataset
    if(collapse_sets):
        temp = temp.groupby("set").mean().reset_index() #collapsing subject by set
    sns.displot(x=x,data=temp)

def bin_by_feature_correlation (dataset, x,y="rate shifted - rate swapped (NN)",bins=3,diatonic="include",normalize=True, figsize=(6,6)):
    temp = dataset #loading dataset
    temp = temp.groupby("set").mean().reset_index() #collapsing subject by set
    if(diatonic=="exclude"): temp = temp[temp['subset_of_diatonic']==False]
    elif(diatonic=="only"): temp = temp[temp['subset_of_diatonic']==True]

    if(normalize):
        min = temp[x].min()
        max = temp[x].max()
        temp[x] = (temp[x]-min)/(max-min)

    temp[x+"_binned"] = pd.qcut(temp[x],bins,precision=2,duplicates="drop") # cut the data into bins

    temp = temp.groupby(x+"_binned").mean().reset_index()

    temp = temp.dropna(subset=[x])
    #plot the data


    fig, ax = plt.subplots(figsize=figsize)

    sns.regplot(x=x, y=y, data=temp, ax=ax)
    annotate(ax,x=x, y=y, data=temp)
    plt.show()

def correlation (dataset, x,y="rate shifted - rate swapped (NN)",diatonic="include", normalize=True, figsize=(6,6), type='pearson', itirations = 10000, show_stats=True, width=8, height=6, save_to=None):
    temp = dataset #loading dataset
    temp = temp.groupby("set").mean(numeric_only=True).reset_index() #collapsing subject by set
    if(diatonic=="exclude"): temp = temp[temp['subset_of_diatonic']==False]
    elif(diatonic=="only"): temp = temp[temp['subset_of_diatonic']==True]

    if(normalize):
        min = temp[x].min()
        max = temp[x].max()
        temp[x] = (temp[x]-min)/(max-min)

    fig, ax = plt.subplots(figsize=figsize)

    #plot the data
    fig.set_size_inches(width, height)

    sns.regplot(x=x, y=y, data=temp, ax=ax)
    if show_stats:
        annotate(ax,x=x, y=y, data=temp, type=type,itirations=itirations)

    if save_to:
        plt.savefig(save_to)


def ridge_coeffs(dataset, X_vars,y_vars,diatonic="include"):

    dataset = dataset.groupby("set").mean().reset_index()  # collapsing subject by set

    if (diatonic == "exclude"): dataset = dataset[dataset['subset_of_diatonic'] == False]
    elif (diatonic == "only"):
        dataset = dataset[dataset['subset_of_diatonic'] == True]


    score, coefficients = ridge_regression(dataset=dataset, X_vars=X_vars, y_vars=y_vars)
    print("****Ridge regression:***")
    print("Score: {}\n coeffs: \n{}".format(score, coefficients))

    print("\nComputing confidence intervals.")
    print(X_vars)
    ci_interval, ci_upper, ci_lower = bootstrap_ci(dataset=dataset, X_vars=X_vars, y_vars=y_vars)
    print("DONE.")

    print("\nRunning Permutation Tests.")
    coefs, P_Vals = permtest_coeffs(X_vars=X_vars, y_vars=y_vars, coefficients=coefficients, dataset=dataset)
    print("p-values:")
    print(P_Vals)
    print("DONE.")

    print("\nPlotting:")
    fig1, ax = plt.subplots(figsize=(8, 6))
    sns.set(font="Arial")
    sns.set_theme(style="white")
    sns.despine()
    ax = sns.barplot(x="vars", y="Coefficient importance", data=coefs, palette=["#b2abd2", "#b2abd2", "#b2abd2"])
    if(ci_interval is not None):
        ax.errorbar(range(len(X_vars)), coefs["Coefficient importance"],
                    yerr=ci_interval,
                    fmt='.', mfc='black', mec='black', ms=0, color='k', linewidth=2)
    ax.set_yticklabels(np.round(ax.get_yticks(), 3), size=13)
    sns.despine()
    ax.axhline(0, color='k', linewidth=.8)
    plt.xlabel("Features", fontsize=13)
    plt.ylabel("Beta Coefficient", fontsize=13)
    # add 95% confidence intervals from bootstrap procedure
    # plt.show()









