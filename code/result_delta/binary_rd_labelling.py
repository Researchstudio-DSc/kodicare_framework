"""
Implement result delta labels using binary labels
"""

import pandas as pd

from code.data import evolving_dtc_splits_parser
from code.result_delta.result_delta_labeling_interface import *
from code.utils.io_util import *


class BinaryRDLabelling(ResultDeltaLabelingInterface):
    def __init__(self, evaluation_dir, epochs, start_step, end_step, systems, metrics):
        self.evaluation_dir = evaluation_dir
        self.epochs = epochs
        self.start_step = start_step
        self.end_step = end_step
        self.systems = systems
        self.metrics = metrics
        self.tc_system_metric_res_maps = self.init_tc_system_metric_res_maps()

    def construct_rd_labels(self):
        res_change_map = self.init_rd_map()
        normal_dist_pairs = not_normal_dist_pairs = 0

        for step in range(self.start_step, self.end_step):
            for tc_ind in range(0, self.epochs - step):
                for tc_ind2 in range(tc_ind + step, self.epochs - step):
                    print('RD for', tc_ind, tc_ind2, step)
                    if tc_ind2 >= self.epochs:
                        break
                    self.update_res_change_map(res_change_map, tc_ind, tc_ind + step, tc_ind2,
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

    def init_rd_map(self):
        res_change_map = {}
        for system in self.systems:
            for metric in self.metrics:
                res_change_map[system + '-' + metric] = []
        return res_change_map

    def init_tc_system_metric_res_maps(self):
        tc_system_metric_res_maps = []
        for i in range(self.epochs):
            tc_system_metric_res_maps.append(self.construct_system_metric_res_map(i))
        return tc_system_metric_res_maps

    def construct_system_metric_res_map(self, tc_id):
        results_df = pd.read_csv(join(self.evaluation_dir, 'tc_' + str(tc_id) + '.csv'), sep='\t')
        system_metric_res_map = self.init_system_metric_res_map()
        for i in range(len(results_df)):
            for system in self.systems:
                system_metric_res_map[system][results_df['metric'][i]].append(results_df[system][i])
        return system_metric_res_map

    def init_system_metric_res_map(self):
        system_metric_res_map = {}
        for system in self.systems:
            system_metric_res_map[system] = {}
            for metric in self.metrics:
                system_metric_res_map[system][metric] = []
        return system_metric_res_map

    def update_res_change_map(self, res_change_map, interval_start1, interval_end1, interval_start2,
                              interval_end2, normal_dist_pairs, not_normal_dist_pairs):
        for system in self.systems:
            for metric in self.metrics:
                acc_res_1 = []  # will contain results for topics for each test collection in the first interval
                acc_res_2 = []  # will contain results for topics for each test collection in the second interval

                for i in range(interval_start1, interval_end1):
                    acc_res_1 += self.tc_system_metric_res_maps[i][system][metric]

                for i in range(interval_start2, interval_end2):
                    acc_res_2 += self.tc_system_metric_res_maps[i][system][metric]

                # TODO: add here test for normal distribution
                if evolving_dtc_splits_parser.is_data_normal(acc_res_1) and evolving_dtc_splits_parser.is_data_normal(
                        acc_res_2):
                    normal_dist_pairs += 1
                else:
                    not_normal_dist_pairs += 1

                res_change_map[system + '-' + metric].append(
                    evolving_dtc_splits_parser.is_significant_change(acc_res_1, acc_res_2))
