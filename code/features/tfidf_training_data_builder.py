"""
implement training data builder for the tfidf representation
"""

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm

from code.data import evolving_dtc_splits_parser
from code.features import training_data_builder_interface
from code.utils import io_util

START_STEP = 2
END_STEP = 10

SYSTEMS = ['bm25_qe_run', 'bm25_run', 'dirLM_qe_run', 'dirLM_run', 'dlh_qe_run', 'dlh_run', 'pl2_qe_run', 'pl2_run']
METRICS = ['P_10', 'Rprec', 'bpref', 'map', 'ndcg', 'ndcg_cut_10', 'recip_rank']


def construct_tc_rep(vocab_dict, doc_vectors):
    tc_rep = np.zeros((len(doc_vectors), len(vocab_dict)))
    for doc_ind, doc_vec in enumerate(doc_vectors):
        for k, v in doc_vec:
            tc_rep[doc_ind][k] = v

    return tc_rep


def vec_diff(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def init_rd_map():
    res_change_map = {}
    for system in SYSTEMS:
        for metric in METRICS:
            res_change_map[system + '-' + metric] = []
    return res_change_map


def update_res_change_map(res_change_map, evolving_dtc_parser, interval_start1, interval_end1, interval_start2,
                          interval_end2, normal_dist_pairs, not_normal_dist_pairs):
    for system in SYSTEMS:
        for metric in METRICS:
            acc_res_1 = []  # will contain results for topics for each test collection in the first interval
            acc_res_2 = []  # will contain results for topics for each test collection in the second interval

            for i in range(interval_start1, interval_end1):
                acc_res_1 += evolving_dtc_parser.get_collection_runs_evaluation_list(system, 'dtc_evolving_eval',
                                                                                     metric, i)
            for i in range(interval_start2, interval_end2):
                acc_res_2 += evolving_dtc_parser.get_collection_runs_evaluation_list(system, 'dtc_evolving_eval',
                                                                                     metric, i)
            # print(acc_res_1)
            # print(acc_res_2)
            # TODO: add here test for normal distribution
            if evolving_dtc_splits_parser.is_data_normal(acc_res_1) and evolving_dtc_splits_parser.is_data_normal(acc_res_2):
                normal_dist_pairs += 1
            else:
                not_normal_dist_pairs += 1
                if not evolving_dtc_splits_parser.is_data_normal(acc_res_1):
                    print('Result interval for', interval_start1, interval_end1, 'not normal')
                if not evolving_dtc_splits_parser.is_data_normal(acc_res_2):
                    print('Result interval for', interval_start2, interval_end2, 'not normal')

            res_change_map[system + '-' + metric].append(
                evolving_dtc_splits_parser.is_significant_change(acc_res_1, acc_res_2))


class TFIDFTrainingDataBuilder(training_data_builder_interface.TrainingDataBuilderInterface):
    def __init__(self, dtc_dir, evaluation_file, representation_dir, epochs=41):
        self.dtc_dir = dtc_dir
        self.evaluation_file = evaluation_file
        self.representation_dir = representation_dir
        self.epochs = epochs

    def build_kd_feature_vector(self):
        vocab_dict = io_util.read_pickle(io_util.join(self.dtc_dir, self.representation_dir + "/vocab.pkl"))
        diff_vectors = []
        concat_feature_df = pd.DataFrame()
        for step in range(START_STEP, END_STEP):
            for tc_ind in range(0, self.epochs - step):
                for tc_ind2 in range(tc_ind + step, self.epochs - step):
                    print('kd for', tc_ind, tc_ind2, step)
                    if tc_ind2 >= self.epochs:
                        break
                    vec1 = io_util.read_pickle(
                        io_util.join(self.dtc_dir, self.representation_dir + '/' + str(tc_ind) + '_tfidf_full_vec.pkl'))
                    vec2 = io_util.read_pickle(io_util.join(self.dtc_dir, self.representation_dir + '/' + str(
                        tc_ind2) + '_tfidf_full_vec.pkl'))
                    tc_rep1 = construct_tc_rep(vocab_dict, vec1)
                    tc_rep2 = construct_tc_rep(vocab_dict, vec2)
                    # get the difference between representation of the same word between two different test collections
                    voc_len = len(vocab_dict)
                    diag_vec = np.zeros(voc_len)
                    for voc_ind in range(voc_len):
                        vec_length = min(len(tc_rep1[:, voc_ind]), len(tc_rep2[:, voc_ind]))
                        diag_vec[voc_ind] = vec_diff(tc_rep1[:, voc_ind][:vec_length], tc_rep2[:, voc_ind][:vec_length])
                    diff_vectors.append(diag_vec)

            features_df = pd.DataFrame(diff_vectors, columns=[vocab_dict[key] for key in vocab_dict.keys()])
            tc_names = self.get_tc_names()
            features_df['tc_name'] = tc_names

            # add some calculations
            kd_norm = np.zeros(len(tc_names))
            kd_min = np.zeros(len(tc_names))
            kd_max = np.zeros(len(tc_names))
            kd_avg = np.zeros(len(tc_names))

            for i in range(len(tc_names)):
                data = diff_vectors[i]
                new_data = data[~np.isnan(data)]
                kd_norm[i] = norm(new_data)
                kd_min[i] = np.min(new_data)
                kd_max[i] = np.max(new_data)
                kd_avg[i] = np.average(new_data)
            features_df['kd_norm'] = kd_norm
            features_df['kd_min'] = kd_min
            features_df['kd_max'] = kd_max
            features_df['kd_avg'] = kd_avg
            features_df.fillna(0, inplace=True)
            features_df.to_pickle(io_util.join(dtc_dir, 'tfidf_eval0_' + str(step) + '.pkl'))
            if concat_feature_df.empty:
                concat_feature_df = features_df
            else:
                concat_feature_df = pd.concat([concat_feature_df, features_df], ignore_index=True)

        return concat_feature_df

    def build_rd_df(self):
        res_change_map = init_rd_map()
        evaluation_splits_path = io_util.join(self.dtc_dir, self.evaluation_file)
        evolving_dtc_parser = evolving_dtc_splits_parser.EvolvingDTCSplitsParser(evaluation_splits_path)
        normal_dist_pairs, not_normal_dist_pairs = 0

        for step in range(START_STEP, END_STEP):
            for tc_ind in range(0, self.epochs - step):
                for tc_ind2 in range(tc_ind + step, self.epochs - step):
                    print('RD for', tc_ind, tc_ind2, step)
                    if tc_ind2 >= self.epochs:
                        break
                    update_res_change_map(res_change_map, evolving_dtc_parser, tc_ind, tc_ind + step, tc_ind2,
                                          tc_ind2 + step, normal_dist_pairs, not_normal_dist_pairs)

        rd_df = pd.DataFrame.from_dict(res_change_map)
        rd_df['tc_name'] = self.get_tc_names()
        print('# of normal distributions:', normal_dist_pairs)
        print('# of not normal distributions:', not_normal_dist_pairs)

        return rd_df

    def get_tc_names(self):
        tc_names = []
        for step in range(START_STEP, END_STEP):
            for tc_ind in range(0, self.epochs - step):
                for tc_ind2 in range(tc_ind + step, self.epochs - step):
                    tc_names.append(
                        str(tc_ind) + '-' + str(tc_ind + step) + ':' + str(tc_ind2) + '-' + str(tc_ind2 + step))
        return tc_names
