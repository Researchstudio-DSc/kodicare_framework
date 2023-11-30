"""
python script to simulate the creation of Evolving Evaluation Environments (EvEE)
"""
import pandas as pd

from code.utils.io_util import *


def get_docno_list(csv_path):
    data = pd.read_csv(csv_path)
    return data['docno'].tolist()


class EvEESimulator:
    def __init__(self, docs_info_path, n_evee, overlap_percentage, output_path):
        """
        the class constructor
        :param docs_info_path: csv file contains at least docno (id of documents) and expected to be ordered by date
        :param n_evee: number of EE to create
        :param overlap_percentage: the percentage of overlap between EE
        :param output_path: the json output path of EEs
        :return:
        """
        self.docs_info_path = docs_info_path
        self.n_evee = n_evee
        self.overlap_percentage = overlap_percentage
        self.output_path = output_path

    def simulate_evee(self):
        docnos = get_docno_list(self.docs_info_path)
        n_docs = len(docnos)
        n_overlap = int((n_docs / self.n_evee) * (self.overlap_percentage / 100))
        print("overlap number", n_overlap)
        n_docs_per_collection = int((n_docs / self.n_evee)) + n_overlap
        evee = {}
        start_index = 0
        end_index = n_docs_per_collection
        for i in range(self.n_evee):
            ee = docnos[start_index:end_index]
            evee[str(i)] = ee
            start_index = end_index - n_overlap
            end_index = min(n_docs, end_index + n_docs_per_collection - n_overlap)
        write_json(self.output_path, evee)
