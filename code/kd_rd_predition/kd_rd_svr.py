"""
Build classification model using SVM to predict binary RD given KD
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, max_error, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.svm import SVR

WEIGHTS_PERMUTATIONS = [
    # 1- one feature only
    (0, 0, 6),
    (0, 6, 0),
    (6, 0, 0),

    # 2- consider two features with equal weights
    (0, 3, 3),
    (3, 3, 0),
    (3, 0, 3),

    # 3- the three features with equal weights/contribution
    (2, 2, 2),

    # 4- all features but one is more dominant than the other two
    (4, 1, 1),
    (1, 4, 1),
    (1, 1, 4),
]


def build_training_features_df_tfidf(weights, rows_numbers, diff_vectors_l, diff_vectors_m, diff_vectors_u):
    features_list = []
    for row_ind in range(rows_numbers):
        features_list.append(np.add(weights[0] * diff_vectors_l[row_ind],
                                    np.add(weights[1] * diff_vectors_m[row_ind],
                                           weights[2] * diff_vectors_u[row_ind])))
    return features_list


def classify_cross_validation_tfidf(diff_vectors_l, diff_vectors_m, diff_vectors_u, vocab_dict, rd_labels_df,
                                    plot_data_path_prefix, systems, metrics):
    for sys in systems:
        mse_map = {}
        me_map = {}
        mae_map = {}
        eve_map = {}
        mse_std_map = {}
        me_std_map = {}
        mae_std_map = {}
        eve_std_map = {}

        for metric in metrics:
            mse_map[metric] = []
            mae_map[metric] = []
            eve_map[metric] = []
            me_map[metric] = []
            mse_std_map[metric] = []
            mae_std_map[metric] = []
            eve_std_map[metric] = []
            me_std_map[metric] = []

        for weights in WEIGHTS_PERMUTATIONS:
            features_list = build_training_features_df_tfidf(weights, len(diff_vectors_l), diff_vectors_l,
                                                             diff_vectors_m, diff_vectors_u)
            features_df = pd.DataFrame(features_list, columns=[vocab_dict[key] for key in vocab_dict.keys()])
            for metric in metrics:
                print('Running SVR CV for system', sys, 'and metric', metric)
                clf = SVR()
                scoring = {'mse': make_scorer(mean_squared_error),
                           'mae': make_scorer(mean_absolute_error),
                           'eve': make_scorer(explained_variance_score),
                           'me': make_scorer(max_error)}
                scores = cross_validate(clf, features_df.head(600), rd_labels_df[sys + '-' + metric].head(600),
                                        scoring=scoring, cv=5, return_train_score=True)

                mse_map[metric].append(round(scores['test_mse'].mean(), 3))
                mae_map[metric].append(round(scores['test_mae'].mean(), 3))
                eve_map[metric].append(round(scores['test_eve'].mean(), 3))
                me_map[metric].append(round(scores['test_me'].mean(), 3))
                mse_std_map[metric].append(round(scores['test_mse'].std(), 3))
                mae_std_map[metric].append(round(scores['test_mae'].std(), 3))
                eve_std_map[metric].append(round(scores['test_eve'].std(), 3))
                me_std_map[metric].append(round(scores['test_me'].std(), 3))

        # save plot data
        plotdata_acc = pd.DataFrame(mse_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_acc_std = pd.DataFrame(mse_std_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_acc.to_csv(plot_data_path_prefix + '_mse_cv_' + sys + '.csv', sep="\t")
        plotdata_acc_std.to_csv(plot_data_path_prefix + '_mse_std_cv_' + sys + '.csv', sep="\t")

        plotdata_p = pd.DataFrame(mae_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_p_std = pd.DataFrame(mae_std_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_p.to_csv(plot_data_path_prefix + '_mae_cv_' + sys + '.csv', sep="\t")
        plotdata_p_std.to_csv(plot_data_path_prefix + '_mae_std_cv_' + sys + '.csv', sep="\t")

        plotdata_r = pd.DataFrame(eve_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_r_std = pd.DataFrame(eve_std_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_r.to_csv(plot_data_path_prefix + '_eve_cv_' + sys + '.csv', sep="\t")
        plotdata_r_std.to_csv(plot_data_path_prefix + '_eve_std_cv_' + sys + '.csv', sep="\t")

        plotdata_f = pd.DataFrame(me_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_f_std = pd.DataFrame(me_std_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_f.to_csv(plot_data_path_prefix + '_me_cv_' + sys + '.csv', sep="\t")
        plotdata_f_std.to_csv(plot_data_path_prefix + '_me_std_cv_' + sys + '.csv', sep="\t")
