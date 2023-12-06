"""
implement the result delta label interface by calculating the absolute difference between the results from two EEs
"""
import pandas as pd

from code.result_delta.result_delta_labeling_interface import ResultDeltaLabelingInterface


class AbsoluteRDLabelling(ResultDeltaLabelingInterface):
    def __init__(self, ee_results_path_1, ee_results_path_2, output_path):
        """
        constructor
        :param ee_results_path_1: the csv path of results from runs on EE at time T1
        :param ee_results_path_2: the csv path of results from runs on EE at time T2
        :param output_path: the csv path of the output
        """
        self.df_results_1 = pd.read_csv(ee_results_path_1, sep='\t')
        self.df_results_2 = pd.read_csv(ee_results_path_2, sep='\t')
        self.output_path = output_path

    def construct_rd_labels(self):
        """
        function to calculate and construct labels for the results delta between systems
        :return:
        """
        df_rd = pd.DataFrame().reindex_like(self.df_results_1)
        for col in self.df_results_1.columns:
            for i in range(len(self.df_results_1)):
                if col == 'Unnamed: 0' or col == 'qid' or col == 'metric':
                    df_rd[col][i] = self.df_results_1[col][i]
                else:
                    df_rd[col][i] = self.df_results_1[col][i] - self.df_results_2[col][i]
        df_rd.to_csv(self.output_path, sep='\t')
