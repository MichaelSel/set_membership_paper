#############################
# Import dependencies
#############################
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import absolute
from numpy import mean
from numpy import std


def fit_data(dataset, X_vars = ['IC1','IC2','IC3','IC4','IC5', 'IC6'], Y_vars = 'rate shifted - rate swapped (NN)', saveDir = './Figures/Behavioral_summaryFigs/',SaveFigs = False, verbose=True):
    #############################################
    # Initializing vars and loading data
    #############################################




    # load data as panda structure
    measures = dataset



    #############################
    # Define measures (Model 1 containing all parameters; word freq and nll)
    #############################
    X = measures[X_vars].astype(float)
    y = measures[Y_vars].to_numpy()

    # reshape if single feature
    if len(X_vars) == 1:
        X = X.to_numpy()
        X = X.reshape(-1, 1)

    # #############################
    # # Multiple regression model
    # #############################
    # # splitting the data
    # # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # # create an object of LinearRegression class
    # LR = LinearRegression()
    # # fit training data
    # LR.fit(X, y)
    # # predict on test data
    # y_prediction = LR.predict(X)
    # # predict the accuracy score (r2)
    # score = r2_score(y, y_prediction)
    # # get model coefficients
    # coefficients = LR.coef_ * X.std(axis=0)



    clf = Ridge(alpha=1.0)
    clf.fit(X, y)
    y_prediction = clf.predict(X)
    score = r2_score(y, y_prediction)
    coefficients = clf.coef_ * X.std(axis=0)

    #########################################
    # Print results
    #########################################
    if(verbose):
        print("##### Model stats: #####")
        print('r2 score is', round(score, ndigits=3))
        print('mean_sqrd_error is', round(mean_squared_error(y, y_prediction), ndigits=3))
        print('root_mean_squared error of is', round(np.sqrt(mean_squared_error(y, y_prediction)), ndigits=3))
        print('residual sum of squares is : ' + str(round(np.sum(np.square(y - y_prediction)), ndigits=3)))

    ##############################################################
    # Compute bootstrapped confidence intervals over coefficients
    ##############################################################
    alist = [[] for _ in range(len(X_vars))]
    measures_bootstrap = np.empty(len(X_vars), object)
    measures_bootstrap[:] = alist
    n_iter_boot = 1000
    for it in range(n_iter_boot):
        # np.random.shuffle(y_rand)
        samples = np.random.choice(len(y), size=len(y), replace=True)
        # grab 80% of the data at random
        X_boot = X.iloc[samples, :]
        Y_boot = y[samples]
        # splitting the data
        # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # create an object of LinearRegression class
        LR = LinearRegression()
        # fit training data
        LR.fit(X_boot, Y_boot)
        # predict on test data
        y_prediction = LR.predict(X_boot)
        # predict the accuracy score (r2)
        score = r2_score(Y_boot, y_prediction)
        # get model coefficients
        boot_coefficients = LR.coef_ * X_boot.std(axis=0)
        #measures_bootstrap[it] = boot_coefficients
        # save null values
        for ii, var in enumerate(X_vars):
            if len(X_vars) == 1:
                measures_bootstrap[ii].append(boot_coefficients)
            else:
                measures_bootstrap[ii].append(eval("boot_coefficients." + var))

    # save bootstrap as panda
    bootstrap_panda = pd.DataFrame()
    for ii, var in enumerate(X_vars):
        bootstrap_panda[X_vars[ii]] = measures_bootstrap[ii]

    bootstrap_panda = bootstrap_panda[bootstrap_panda[:] < 1]
    bootstrap_panda = bootstrap_panda[bootstrap_panda[:] > -1]
    bootstrap_panda = bootstrap_panda.dropna(how = 'all')

    # generate confidence intervals
    upper = np.percentile(bootstrap_panda, 97.5, axis=0)
    lower = np.percentile(bootstrap_panda, 2.5, axis=0)
    interval = np.percentile(bootstrap_panda, 97.5, axis=0)-np.mean(bootstrap_panda, axis=0)


    #########################################
    # Run perm. tests for coeffs.
    #########################################
    n_iter = 10000
    if len(X_vars) == 1:
        B = coefficients
    else:
        B = coefficients.array

    coefs = pd.DataFrame(
        B,
        columns=["Coefficient importance"],
        index=X_vars,
    )
    alist = [[] for _ in range(len(X_vars))]
    measures_null = np.empty(len(X_vars), object)
    measures_null[:] = alist
    y_rand = y
    for it in range(n_iter):
        np.random.shuffle(y_rand)
        # splitting the data
        # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # create an object of LinearRegression class
        LR = LinearRegression()
        # fit training data
        LR.fit(X, y)
        # predict on test data
        y_prediction = LR.predict(X)
        # predict the accuracy score (r2)
        score = r2_score(y, y_prediction)
        # get model coefficients
        rand_coefficients = LR.coef_ * X.std(axis=0)
        # save null values
        for ii, var in enumerate(X_vars):
            if len(X_vars) == 1:
                measures_null[ii].append(rand_coefficients)
            else:
                measures_null[ii].append(eval("rand_coefficients." + var))

    # Two-sided permutation test
    _ = plt.hist(np.abs(measures_null[0]), bins='auto')
    plt.show()
    P_Vals = []
    for ii, var in enumerate(X_vars):
        if len(X_vars) == 1:
            P_Vals.append(np.mean(np.abs(measures_null[ii]) > np.abs(B), axis=0))
        else:
            P_Vals.append(np.mean(np.abs(measures_null[ii]) > np.abs(eval("coefficients." + var)), axis=0))
    # store p-values
    coefs['vars'] = X_vars

    ################################################
    # SUPP Fig 2: Plot coefficients with confidence intervals
    ################################################
    # construct pandas for plotting
    coefs = pd.DataFrame(data={'vars':X_vars, "Coefficient importance":coefficients})
    # plotting
    fig1,ax = plt.subplots(figsize=(8, 6))
    sns.set(font="Arial")
    sns.set_theme(style="white")
    sns.despine()
    ax = sns.barplot(x="vars", y="Coefficient importance", data=coefs,palette=["#b2abd2", "#b2abd2", "#b2abd2"])
    ax.errorbar(range(len(X_vars)), coefs["Coefficient importance"],
                 yerr = interval,
                 fmt ='.', mfc='black', mec='black', ms=0, color='k', linewidth = 2)
    ax.set_yticklabels(np.round(ax.get_yticks(),3), size = 13)
    sns.despine()
    ax.axhline(0, color='k', linewidth=.8)
    plt.xlabel("Scale Intervals", fontsize=13)
    plt.ylabel("Beta Coefficient", fontsize=13)
    # add 95% confidence intervals from bootstrap procedure
    plt.show()

    if SaveFigs:
        fig1.savefig(saveDir+'interval_coefficients.svg')



    #############################
    # SUPP Fig 1: Visualize feature space measures
    #############################
    All_vars = measures[['IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'IC6']].astype(float)
    # pairplot with colors
    bright = '#d8daeb'
    dark = '#b2abd2'


    sns.set(font="Arial")
    sns.set_theme(style="white")
    sns.despine()
    g = sns.PairGrid(All_vars)
    plt.xlabel("Scale Intervals", fontsize=13)
    plt.ylabel("Beta Coefficient", fontsize=13)
    plt.show()
    if SaveFigs:
        plt.savefig(saveDir + "Features_PairPlot_frame.svg")

    # feature correlation plot
    fig2 = plt.figure()
    sns.set(font="Arial")
    sns.set_theme(style="white")
    sns.despine()
    sns.heatmap(X.corr(),annot=True,lw=1)
    plt.show()
    if SaveFigs:
        fig2.savefig(saveDir + "Feature_Corr_Matrix.tiff")