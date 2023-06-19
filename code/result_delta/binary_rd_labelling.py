"""
Implement result delta labels using binary labels
"""

import pandas as pd

from code.data import evolving_dtc_splits_parser
from code.result_delta import result_delta_labeling_interface
from code.utils import io_util

SYSTEMS = ['bm25_qe_run', 'bm25_run', 'dirLM_qe_run', 'dirLM_run', 'dlh_qe_run', 'dlh_run', 'pl2_qe_run', 'pl2_run']
METRICS = ['P_10', 'Rprec', 'bpref', 'map', 'ndcg', 'ndcg_cut_10', 'recip_rank']


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
            # TODO: add here test for normal distribution
            if evolving_dtc_splits_parser.is_data_normal(acc_res_1) and evolving_dtc_splits_parser.is_data_normal(
                    acc_res_2):
                normal_dist_pairs += 1
            else:
                not_normal_dist_pairs += 1

            res_change_map[system + '-' + metric].append(
                evolving_dtc_splits_parser.is_significant_change(acc_res_1, acc_res_2))


class BinaryRDLabelling(result_delta_labeling_interface.ResultDeltaLabelingInterface):
    def __init__(self, dtc_dir, evaluation_file, epochs, start_step, end_step):
        self.dtc_dir = dtc_dir
        self.evaluation_file = evaluation_file
        self.epochs = epochs
        self.start_step = start_step
        self.end_step = end_step

    def construct_rd_labels(self):
        res_change_map = init_rd_map()
        evaluation_splits_path = io_util.join(self.dtc_dir, self.evaluation_file)
        evolving_dtc_parser = evolving_dtc_splits_parser.EvolvingDTCSplitsParser(evaluation_splits_path)
        normal_dist_pairs = not_normal_dist_pairs = 0

        for step in range(self.start_step, self.end_step):
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
        for step in range(self.start_step, self.end_step):
            for tc_ind in range(0, self.epochs - step):
                for tc_ind2 in range(tc_ind + step, self.epochs - step):
                    tc_names.append(
                        str(tc_ind) + '-' + str(tc_ind + step) + ':' + str(tc_ind2) + '-' + str(tc_ind2 + step))
        return tc_names
