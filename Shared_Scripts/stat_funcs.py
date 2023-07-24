# Set of functions for statistics. Main function to perform permutation tests for various statistical comparisons
from statistics import stdev

import pandas as pd
from numpy import std, mean, sqrt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from statsmodels.stats.anova import AnovaRM
import numpy as np
from scipy import stats
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from scipy import spatial
import pingouin as pg

# Function to shuffle contents of a Panda structure
def shuffle_panda(df, n, axis=0):
    shuffled_df = df.copy()
    for k in range(n):
        shuffled_df.apply(np.random.shuffle(shuffled_df.values), axis=axis)
    return shuffled_df


# function to run permutation test for a variety of statistical comparisons. BehavMeasure ('RT' or 'PC')
def permtest_ANOVA_paired(data_panda, behavMeasure, reps):
    # initialize vector to hold statistic on each iteration
    rand_vals = list()

    # get observed statistics (interaction) for two-way ANOVA
    aovrm2way = AnovaRM(data_panda, behavMeasure, 'Subject_ID', within=['task', 'condition'])
    results_table = aovrm2way.fit()
    F_vals = results_table.anova_table['F Value']

    # get observed interaction F-value: condition-task
    obs_stat = F_vals[2]

    # deep copy of panda structure
    shuffled_panda = data_panda.copy()

    # loop through repetitions
    for ii in range(reps):
        print('\r{} of {}'.format(ii, reps), end='')

        # shuffle column with behavioral measure of interest (RT or PC)
        shuffled_panda[behavMeasure] = np.random.permutation(shuffled_panda[behavMeasure].values)

        # get randomized statistic (interaction) for two-way ANOVA
        aovrm2way_rand = AnovaRM(shuffled_panda, behavMeasure, 'Subject_ID', within=['task', 'condition'])
        results_table_rand = aovrm2way_rand.fit()
        F_vals_rand = results_table_rand.anova_table['F Value']

        # get interaction F-value for shuffled structure: condition-task
        rand = F_vals_rand[2]

        # push back rand F value
        rand_vals.append(rand)

    rand_vals = np.array(rand_vals)

    # look at probability on either side of the distribution based on the observed statistic - this function is
    # therefore order invariant with respect to its inputs
    prob = np.mean(np.abs(rand_vals) > np.abs(obs_stat))

    _ = plt.hist(rand_vals, bins='auto')  # arguments are passed to np.histogram
    plt.show()

    print(f'p = {prob}')
    print(f'obs_stat = {obs_stat}')

    return obs_stat, prob

def permtest_coeffs(X_vars,y_vars,coefficients,dataset, n_iter = 10000, plot=False):

    if len(X_vars) == 1:
        B = coefficients
    else:
        B = coefficients.array

    X = dataset[X_vars].astype(float)
    y = dataset[y_vars].to_numpy()

    # reshape if single feature
    if len(X_vars) == 1:
        X = X.to_numpy()
        X = X.reshape(-1, 1)

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
        print('\r{} of {}'.format(it, n_iter), end='')
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
    if(plot):
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
    coefs = pd.DataFrame(data={'vars': X_vars, "Coefficient importance": coefficients})
    return coefs

# function to run permutation test for a pearson correlation
def perm_t_test_unpaired(X, Y, reps):
    # convert input to numpy
    X = np.array(X)
    Y = np.array(Y)

    # initialize vector to hold statistic on each iteration
    rand_vals = list()

    # get observed statistic based on the test of interest
    observation = stats.ttest_ind(X, Y)
    obs_stat = observation[0]

    # concatenate data from both vars
    data_concat = np.concatenate((X, Y), axis=0)

    for ii in range(reps):
        print('\r{} of {}'.format(ii, reps), end='')

        # shuffle data and split into two random groups
        np.random.shuffle(data_concat)
        random_split = np.array_split(data_concat, 2)

        rand = stats.ttest_ind(random_split[0], random_split[1])
        rand = rand[0]

        # push back R value
        rand_vals.append(rand)

    rand_vals = np.array(rand_vals)

    # look at probability on either side of the distribution based on the observed statistic - this function is
    # therefore order invariant with respect to its inputs
    prob = np.mean(np.abs(rand_vals) > np.abs(obs_stat))

    _ = plt.hist(rand_vals, bins='auto')  # arguments are passed to np.histogram
    plt.show()

    print(f'p = {prob}')
    print(f'obs_stat = {obs_stat}')

    return obs_stat, prob


# function to run permutation test for a pearson correlation
def perm_t_test_paired(X, Y, reps):
    # convert input to numpy
    X = np.array(X)
    Y = np.array(Y)

    # initialize vector to hold statistic on each iteration
    rand_vals = list()

    # get observed statistic based on the test of interest
    observation = stats.ttest_rel(X, Y)
    obs_stat = observation[0]

    # concatenate data from both vars
    data_concat = np.concatenate((X, Y), axis=0)

    for ii in range(reps):
        print('\r{} of {}'.format(ii, reps), end='')

        # shuffle data and split into two random groups
        np.random.shuffle(data_concat)
        random_split = np.split(data_concat, 2)

        rand = stats.ttest_rel(random_split[0], random_split[1])
        rand = rand[0]

        # push back R value
        rand_vals.append(rand)

    rand_vals = np.array(rand_vals)

    # look at probability on either side of the distribution based on the observed statistic - this function is
    # therefore order invariant with respect to its inputs
    prob = np.mean(np.abs(rand_vals) > np.abs(obs_stat))

    _ = plt.hist(rand_vals, bins='auto')  # arguments are passed to np.histogram
    plt.show()

    print(f'p = {prob}')
    print(f'obs_stat = {obs_stat}')

    return obs_stat, prob


# function to run permutation test for a cosine similarity
def permtest_cosine_sim(X, Y, reps):
    # convert input to numpy
    X = np.array(X)
    Y = np.array(Y)

    # initialize vector to hold statistic on each iteration
    rand_vals = list()

    # get observed statistic based on the test of interest
    obs_stat = spatial.distance.cosine(X,Y)

    y_shuffled = dc(Y)

    for ii in range(reps):
        print('\r{} of {}'.format(ii, reps), end='')

        np.random.shuffle(y_shuffled)

        rand = spatial.distance.cosine(X, y_shuffled)

        # push back R value
        rand_vals.append(rand)

    rand_vals = np.array(rand_vals)

    # look at probability on either side of the distribution based on the observed statistic (positive/negative
    # correlation)
    prob = np.mean(np.abs(rand_vals) > np.abs(obs_stat))

    _ = plt.hist(rand_vals, bins='auto')
    plt.show()

    print(f'p = {prob}')
    print(f'obs_stat = {obs_stat}')

    return obs_stat, prob


# function to run permutation test for a spearman correlation
def permtest_corr(X, Y, reps):
    # convert input to numpy
    X = np.array(X)
    Y = np.array(Y)

    # initialize vector to hold statistic on each iteration
    rand_vals = list()

    # get observed statistic based on the test of interest
    obs_stat = stats.spearmanr(X, Y)
    obs_stat = obs_stat[0]

    y_shuffled = dc(Y)

    for ii in range(reps):
        print('\r{} of {}'.format(ii, reps), end='')

        np.random.shuffle(y_shuffled)

        rand = stats.spearmanr(X, y_shuffled)

        # push back R value
        rand_vals.append(rand[0])

    rand_vals = np.array(rand_vals)

    # look at probability on either side of the distribution based on the observed statistic (positive/negative
    # correlation)
    prob = np.mean(np.abs(rand_vals) > np.abs(obs_stat))

    # _ = plt.hist(rand_vals, bins='auto')
    # plt.show()

    print(f'p = {prob}')
    print(f'obs_stat = {obs_stat}')

    return obs_stat, prob

# function to run permutation test for a spearman correlation
def permtest_corr_pearson(X, Y, reps,show_hist=True):
    # convert input to numpy
    X = np.array(X)
    Y = np.array(Y)

    # initialize vector to hold statistic on each iteration
    rand_vals = list()

    # get observed statistic based on the test of interest
    obs_stat = stats.pearsonr(X, Y)
    obs_stat = obs_stat[0]

    y_shuffled = dc(Y)

    for ii in range(reps):
        print('\r{} of {}'.format(ii, reps), end='')

        np.random.shuffle(y_shuffled)

        rand = stats.pearsonr(X, y_shuffled)

        # push back R value
        rand_vals.append(rand[0])

    rand_vals = np.array(rand_vals)

    # look at probability on either side of the distribution based on the observed statistic (positive/negative
    # correlation)
    prob = np.mean(np.abs(rand_vals) > np.abs(obs_stat))


    if(show_hist):
        _ = plt.hist(rand_vals, bins='auto')
        plt.show()

    print(f'p = {prob}')
    print(f'obs_stat = {obs_stat}')

    return obs_stat, prob


# function to run permutation test for differences (subtraction between two means for instance)
def perm_bias_paired(X, Y, reps=10000):
    # convert input to numpy
    X = np.array(X)
    Y = np.array(Y)

    # initialize vector to hold statistic on each iteration
    rand_vals = list()

    # get observed statistic based on the test of interest
    observation = X - Y
    obs_stat = np.mean(observation)

    # concatenate data from both vars
    data_concat = np.concatenate((X, Y), axis=0)

    for ii in range(reps):
        print('\r{} of {}'.format(ii, reps), end='')

        # shuffle data and split into two random groups
        np.random.shuffle(data_concat)
        random_split = np.split(data_concat, 2)

        rand = random_split[0] - random_split[1]
        rand = np.mean(rand)
        # push back R value
        rand_vals.append(rand)

    rand_vals = np.array(rand_vals)

    # look at probability on either side of the distribution based on the observed statistic - this function is
    # therefore order invariant with respect to its inputs
    prob = np.mean(np.abs(rand_vals) > np.abs(obs_stat))

    _ = plt.hist(rand_vals, bins='auto')  # arguments are passed to np.histogram
    plt.show()

    print(f'p = {prob}')
    print(f'obs_stat = {obs_stat}')

    return obs_stat, prob

# function to run unpaired permutation test for differences (subtraction between two means for instance)
def perm_bias_unpaired(X, Y, reps=10000, loud=True):
    # convert input to numpy
    X = np.array(X)
    Y = np.array(Y)

    # initialize vector to hold statistic on each iteration
    rand_vals = list()

    # get observed statistic based on the test of interest
    size_x = X.shape[0]
    size_y = Y.shape[0]
    obs_stat = np.mean(X) - np.mean(Y)


    # concatenate data from both vars
    data_concat = np.concatenate((X, Y), axis=0)

    for ii in range(reps):
        if loud:
            print('\r{} of {}'.format(ii, reps), end='')

        # shuffle data and split into two random groups
        np.random.shuffle(data_concat)
        new_X = data_concat[:size_x]
        new_Y = data_concat[-size_y:]

        rand = np.mean(new_X) - np.mean(new_Y)
        # push back R value
        rand_vals.append(rand)

    rand_vals = np.array(rand_vals)

    # look at probability on either side of the distribution based on the observed statistic - this function is
    # therefore order invariant with respect to its inputs
    prob = np.mean(np.abs(rand_vals) > np.abs(obs_stat))
    if loud:
        _ = plt.hist(rand_vals, bins='auto')  # arguments are passed to np.histogram
        plt.show()

        print(f'p = {prob}')
        print(f'obs_stat = {obs_stat}')

    return obs_stat, prob


# Compute cohen's d for unpaired t-test
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx - 1) * std(x) ** 2 + (ny - 1) * std(y) ** 2) / dof)


# Compute cohen's d for paired t-test
def cohen_d_av(x, y):
    return (mean(x) - mean(y)) / ((stdev(x) + stdev(y)) / 2)


def cohen_dz(diff_vector):
    sd = diff_vector.std()
    m = diff_vector.mean()
    return m / sd


def t_value(X, Y):
    return stats.ttest_rel(X, Y)

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

def prime_factor(value):
    factors = []
    for divisor in range(2, value-1):
        quotient, remainder = divmod(value, divisor)
        if not remainder:
            factors.extend(prime_factor(divisor))
            factors.extend(prime_factor(quotient))
            break
        else:
            factors = [value]
    return factors


def perm_bias_unpaired_unpack(args):
    return perm_bias_unpaired(*args)


def bootstrap_ci(X_vars, y_vars, dataset, upper=97.5, lower=2.5):

    if len(X_vars) == 1:
        print("Skipping")
        return None, None, None

    X = dataset[X_vars].astype(float)
    y = dataset[y_vars].to_numpy()

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
        # measures_bootstrap[it] = boot_coefficients
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
    bootstrap_panda = bootstrap_panda.dropna(how='all')

    # generate confidence intervals
    upper_vals = np.percentile(bootstrap_panda, upper, axis=0)
    lower_vals = np.percentile(bootstrap_panda, lower, axis=0)
    interval = np.percentile(bootstrap_panda, upper, axis=0) - np.mean(bootstrap_panda, axis=0)
    return interval, upper_vals, lower_vals

def ridge_regression(dataset, X_vars,y_vars):
    X = dataset[X_vars].astype(float)
    y = dataset[y_vars].to_numpy()

    # reshape if single feature
    if len(X_vars) == 1:
        X = X.to_numpy()
        X = X.reshape(-1, 1)

    clf = Ridge(alpha=1.0)
    clf.fit(X, y)
    y_prediction = clf.predict(X)
    score = r2_score(y, y_prediction)
    coefficients = clf.coef_ * X.std(axis=0)
    return score, coefficients


# function to run permutation test for a variety of statistical comparisons. BehavMeasure ('RT' or 'PC'). stick with
# F-value here. Conds argument needs to be an array of size 1x2 (string labels)
def permtest_ANOVA_mixed(data_panda, dv, between_measure, within_measure, ids, reps):

    # initialize vector to hold statistic on each iteration
    rand_vals = list()

    # get observed statistics (interaction) for two-way ANOVA

    results = pg.mixed_anova(dv=dv, between=between_measure,
                             within=within_measure, subject=ids, data=data_panda)

    # get observed interaction F-value: mixed-effects interaction
    obs_stat = results.F[2]

    # deep copy of panda structure
    shuffled_panda = data_panda.copy()

    # loop through repetitions
    for ii in range(reps):

        print('\r{} of {}'.format(ii, reps), end='')

        # shuffle column with behavioral measure of interest (RT or PC)
        shuffled_panda[dv] = np.random.permutation(shuffled_panda[dv].values)

        # get randomized statistic (interaction) for two-way ANOVA
        results_rand= pg.mixed_anova(dv=dv, between=between_measure,
                                 within=within_measure, subject=ids, data=shuffled_panda)

        # get interaction F-value for shuffled structure: interaction
        F_vals_rand = results_rand.F[2]

        # get interaction F-value for shuffled structure: condition-task
        rand = F_vals_rand

        # push back rand F value
        rand_vals.append(rand)

    rand_vals = np.array(rand_vals)

    # look at probability on either side of the distribution based on the observed statistic - this function is
    # therefore order invariant with respect to its inputs
    prob = np.mean(rand_vals > obs_stat)

    _ = plt.hist(rand_vals, bins='auto')  # arguments are passed to np.histogram
    plt.show()

    print(f'p = {prob}')
    print(f'obs_stat = {obs_stat}')

    return obs_stat, prob