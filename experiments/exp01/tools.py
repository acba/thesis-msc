import numpy as np
import pickle
import os
import datetime
from scipy import stats
import scipy

import pai
from pai.portfolio import get_data


def non_parametric_check_samples(samples, significance=.05):

    # Null hypothesis samples are equals distributions

    # print(samples)

    # Perform 'Kruskal-Wallis H-test' non parametric test
    h_statistic, p_value = stats.kruskal(*samples)

    if p_value < significance:
        return False
    else:
        return True


def parametric_check_samples(samples, significance=.05):

    # Null hypothesis samples are equals distributions

    # Perform 'Anova' parametric test
    f_value, p_value = stats.f_oneway(*samples)

    if p_value < significance:
        return False
    else:
        return True


def check_normality(samples):

    # 5% significance

    result = []
    for sample in samples:

        # Calculate Anderson-Darling normality test index
        ad_statistic, ad_c, ad_s = stats.anderson(sample, "norm")
        if ad_statistic > ad_c[3]:
            result.append(False)
        else:
            result.append(True)

    return result


def matrix_non_parametric_paired_test(samples):

    nparam = np.empty((len(samples), len(samples)))

    for i in range(len(samples)):
        for j in range(len(samples)):
            if i == j:
                nparam[i, j] = False

            else:
                if not non_parametric_check_samples([samples[i], samples[j]]):
                    r = np.mean(samples[i]) > np.mean(samples[j])

                else:
                    r = False
                nparam[i, j] = r

    return nparam


def check(filename):

    if os.path.exists(filename):
        result = pickle.load(open(filename, 'rb'))

        if result["finished"]:
            return False
        else:
            return True

    else:
        return True










# data = [[0.038276776980954046, 0.038278290485972498, 0.038260110575184414,
#          0.038276684460136989, 0.03825459687413521, 0.038278262143575215, 0.038276697554263495, 0.038276848056863502, 0.038299042466280894, 0.038313960261609034, 0.038253778246018617, 0.038277644536465205, 0.038277261948407283, 0.038280358547992763, 0.038276644327923742],
#         [0.038276776980954046, 0.038278290485972498, 0.038260110575184414, 0.038276684460136989, 0.03825459687413521, 0.038278262143575215, 0.038276697554263495, 0.038276848056863502, 0.038299042466280894, 0.038313960261609034, 0.038253778246018617, 0.038277644536465205, 0.038277261948407283, 0.038280358547992763, 0.038276644327923742]]

# data = [[0.038276776980954046, 0.038278290485972498, 0.038260110575184414, 0.038276684460136989, 0.03825459687413521, 0.038278262143575215, 0.038276697554263495, 0.038276848056863502, 0.038299042466280894, 0.038313960261609034, 0.038253778246018617, 0.038277644536465205, 0.038277261948407283, 0.038280358547992763, 0.038276644327923742],
# [0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753, 0.03837291014263753],
# [0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806, 0.03839728835539806]]
#
#
# print(non_parametric_check_samples(data))
# print(parametric_check_samples(data))
# # print(check_normality(data))