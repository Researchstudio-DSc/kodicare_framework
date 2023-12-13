"""
Build classification model using SVM to predict binary RD given KD
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC

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
        acc_map = {}
        f_map = {}
        p_map = {}
        r_map = {}
        acc_std_map = {}
        f_std_map = {}
        p_std_map = {}
        r_std_map = {}

        for metric in metrics:
            acc_map[metric] = []
            p_map[metric] = []
            r_map[metric] = []
            f_map[metric] = []
            acc_std_map[metric] = []
            p_std_map[metric] = []
            r_std_map[metric] = []
            f_std_map[metric] = []

        for weights in WEIGHTS_PERMUTATIONS:
            features_list = build_training_features_df_tfidf(weights, len(diff_vectors_l), diff_vectors_l,
                                                             diff_vectors_m, diff_vectors_u)
            features_df = pd.DataFrame(features_list, columns=[vocab_dict[key] for key in vocab_dict.keys()])
            for metric in metrics:
                clf = SVC()
                scoring = {'acc': make_scorer(accuracy_score),
                           'p': make_scorer(precision_score),
                           'r': make_scorer(recall_score),
                           'f1': make_scorer(f1_score)}
                scores = cross_validate(clf, features_df.head(600), rd_labels_df[sys + '-' + metric].head(600),
                                        scoring=scoring, cv=5, return_train_score=True)

                acc_map[metric].append(round(scores['test_acc'].mean(), 3))
                p_map[metric].append(round(scores['test_p'].mean(), 3))
                r_map[metric].append(round(scores['test_r'].mean(), 3))
                f_map[metric].append(round(scores['test_f1'].mean(), 3))
                acc_std_map[metric].append(round(scores['test_acc'].std(), 3))
                p_std_map[metric].append(round(scores['test_p'].std(), 3))
                r_std_map[metric].append(round(scores['test_r'].std(), 3))
                f_std_map[metric].append(round(scores['test_f1'].std(), 3))

        # save plot data
        plotdata_acc = pd.DataFrame(acc_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_acc_std = pd.DataFrame(acc_std_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_acc.to_csv(plot_data_path_prefix + '_acc_cv_' + sys + '.csv', sep="\t")
        plotdata_acc_std.to_csv(plot_data_path_prefix + '_acc_std_cv_' + sys + '.csv', sep="\t")

        plotdata_p = pd.DataFrame(p_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_p_std = pd.DataFrame(p_std_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_p.to_csv(plot_data_path_prefix + '_p_cv_' + sys + '.csv', sep="\t")
        plotdata_p_std.to_csv(plot_data_path_prefix + '_p_std_cv_' + sys + '.csv', sep="\t")

        plotdata_r = pd.DataFrame(r_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_r_std = pd.DataFrame(r_std_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_r.to_csv(plot_data_path_prefix + '_r_cv_' + sys + '.csv', sep="\t")
        plotdata_r_std.to_csv(plot_data_path_prefix + '_r_std_cv_' + sys + '.csv', sep="\t")

        plotdata_f = pd.DataFrame(f_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_f_std = pd.DataFrame(f_std_map, index=[str(weights) for weights in WEIGHTS_PERMUTATIONS])
        plotdata_f.to_csv(plot_data_path_prefix + '_f_cv_' + sys + '.csv', sep="\t")
        plotdata_f_std.to_csv(plot_data_path_prefix + '_f_std_cv_' + sys + '.csv', sep="\t")
