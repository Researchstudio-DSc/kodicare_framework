"""
class to parse the hdf5 of splits for cord 19 and robust test collection includes
- the generation of ids for dynamic test collection
- the generation of runs results
"""

import h5py
import numpy as np
import ruptures as rpt
from scipy import stats

rng = np.random.default_rng()


class EvolvingDTCSplitsParser:
    SAMPLE_DOCS_HDF5_KEY = 'sample_docs'
    RANDOM_DOCS_HDF5_KEY = 'random_docs'
    PARAMS_HDF5_KEY = 'params'

    def __init__(self, evaluation_splits_file):
        """
        read the hdf5 file
        :param evaluation_splits_file: the file path of the data of the splits and the evaluation results id hdf5 format
        """
        self.evaluation_splits_data = h5py.File(evaluation_splits_file, 'r')

    def create_overlap_sample(self, tc_type='ordered', random_init=True, rm_rate=0):
        """
        Create the Dynamic Test Collection using a ordered list of documents.
        Each environment has a defined overlap with the next one.
        tc_type: {'ordered': consider the date order for the collection creation, 'random': create randomly}
        rm_rate: {0: without removing documents, 1: rem = add, -1: rem = add but random }
        """
        docs_key = self.SAMPLE_DOCS_HDF5_KEY if tc_type == 'ordered' else self.RANDOM_DOCS_HDF5_KEY
        original_docs = self.evaluation_splits_data[docs_key][...]

        # the saved parameters of the splits
        size_port = self.evaluation_splits_data[self.PARAMS_HDF5_KEY][0]
        num_ov = self.evaluation_splits_data[self.PARAMS_HDF5_KEY][1]

        # document sample size.
        ee_size = len(original_docs) // size_port

        # only one overlap: the smaller one.
        ov = ee_size - ee_size // num_ov

        # DTC documents.
        doc_overlaps = {}

        adding_rate = ee_size - ov
        removing_rate = adding_rate

        if rm_rate == 0:
            removing_rate = 0

        if rm_rate == -1:
            # init = 0, remove random docs that belong to the previous version.
            removing_rate = 0

        ## First environment:
        init = 0
        if random_init:
            init = int(np.random.uniform() * ee_size)

        end = init + ee_size
        d0 = original_docs[init:end]

        doc_ees = []
        doc_ees.append(d0)

        ## Next environments
        while (end + adding_rate < len(original_docs)):
            init = init + removing_rate

            if rm_rate == -1:
                ran_docs = doc_ees[-1]

                # are the documents listed in the same order??? no, but it is not important, because they are all in the same ee.
                ran_docs = np.random.choice(doc_ees[-1], size=len(doc_ees[-1]) * (100 - num_ov) // 100, replace=False)

                d0 = np.concatenate((ran_docs, original_docs[end:end + adding_rate]), axis=None)
                end = end + adding_rate

            else:
                end = end + adding_rate
                d0 = original_docs[init:end]

            doc_ees.append(d0)

        doc_overlaps[ov] = {'doc': doc_ees}

        print(len(doc_ees))

        return doc_overlaps

    def get_runs_evaluation(self, run_name, dtc_name, eval_metric, topic_number):
        """
        return the evaluation result of group of dynamic test collection for a specific topic number
        :param run_name: one of these values which refers to the IR system
        {'bm25_qe_run', 'bm25_run', 'dirLM_qe_run', 'dirLM_run', 'dlh_qe_run', 'dlh_run', 'pl2_qe_run', 'pl2_run'}
        :param dtc_name: one of the values refers to the type of DTC creation
        {'dtc_evolving_eval', 'dtc_random_eval', 'stc_random_eval'}
        :param eval_metric: one of the evaluation metrics
        {'P_10', 'Rprec', 'bpref', 'map', 'ndcg', 'ndcg_cut_10', 'recip_rank'}
        :param topic_number: according to the test collection CORD19: from 0-49
        :return: numpy array of evaluation result for each collection
        """
        return self.evaluation_splits_data[run_name][dtc_name][eval_metric][str(topic_number)][...]

    def get_avg_run_evaluation(self, run_name, dtc_name, eval_metric):
        """
        return the average value of all topic results for group of test collection
        :param run_name: one of these values which refers to the IR system
        {'bm25_qe_run', 'bm25_run', 'dirLM_qe_run', 'dirLM_run', 'dlh_qe_run', 'dlh_run', 'pl2_qe_run', 'pl2_run'}
        :param dtc_name: one of the values refers to the type of DTC creation
        {'dtc_evolving_eval', 'dtc_random_eval', 'stc_random_eval'}
        :param eval_metric: one of the evaluation metrics
        {'P_10', 'Rprec', 'bpref', 'map', 'ndcg', 'ndcg_cut_10', 'recip_rank'}
        :return: numpy array of average result
        """
        topic_keys = list(self.evaluation_splits_data[run_name][dtc_name][eval_metric].keys())
        accumulative_results = self.evaluation_splits_data[run_name][dtc_name][eval_metric][topic_keys[0]][...]
        for i in range(1, len(topic_keys)):
            accumulative_results = np.add(
                self.evaluation_splits_data[run_name][dtc_name][eval_metric][topic_keys[i]][...], accumulative_results)
        return accumulative_results / len(topic_keys)

    def get_absolute_results_delta(self, run_name, dtc_name, eval_metric, step):
        """
        returns the result delta as the absolute difference between runs of a dtc
        :param run_name: one of these values which refers to the IR system
        {'bm25_qe_run', 'bm25_run', 'dirLM_qe_run', 'dirLM_run', 'dlh_qe_run', 'dlh_run', 'pl2_qe_run', 'pl2_run'}
        :param dtc_name: one of the values refers to the type of DTC creation
        {'dtc_evolving_eval', 'dtc_random_eval', 'stc_random_eval'}
        :param eval_metric: one of the evaluation metrics
        {'P_10', 'Rprec', 'bpref', 'map', 'ndcg', 'ndcg_cut_10', 'recip_rank'}
        :param step: number of steps to skip between dtcs must be more or equal to 1
        :return: numpy array for calculated RDs
        """
        results = self.get_avg_run_evaluation(run_name, dtc_name, eval_metric)
        results_delta = []
        for tc_ind in range(0, len(results), step):
            if (tc_ind + step) >= len(results):
                break
            results_delta.append(results[tc_ind + step] - results[tc_ind])
        return np.array(results_delta)

    def get_result_delta_change_points(self, results_delta, n_bkps=4):
        """
        returns the change points in the result delta
        :param results_delta: numpy array of results or result delta
        :param n_bkps: number of change points
        :return: a binary numpy array if the value is one then a change happens at the specific point
        """

        change_points = np.zeros(len(results_delta))

        algo = rpt.KernelCPD(kernel="linear", min_size=2).fit(results_delta)
        finished = False
        while not finished:
            try:
                change_points_indices = algo.predict(n_bkps=n_bkps)
                prev_cp_index = 0
                for ind in range(len(change_points_indices) - 1):
                    if self.verify_change_point_significance(
                            results_delta[prev_cp_index:change_points_indices[ind]],
                            results_delta[change_points_indices[ind]:change_points_indices[ind + 1]], p=0.1):
                        change_points[change_points_indices[ind] - 1] = 1
                finished = True
            except rpt.exceptions.BadSegmentationParameters:
                n_bkps -= 1

        return np.array(change_points)

    def verify_change_point_significance(self, interval_1, interval_2, p=0.05):
        # if the mean of the first interval is significantly less than the second then the change is significant
        t_value, p_value = stats.ttest_ind(interval_1, interval_2, alternative='less')
        return p_value < p

    def get_cp_ttest(self, results, step=2, p=0.05):
        change_points = np.zeros(len(results))
        for ind in range(0, len(results) - step, step):
            if (ind + step) >= len(results) or (ind + step * 2) > len(results):
                break
            interval_1 = results[ind:ind + step]
            interval_2 = results[ind + step:ind + step * 2]
            t_value_greater, p_value = stats.ttest_ind(interval_1, interval_2)
            if p_value < p:
                change_points[ind + step] = 1
        return change_points
