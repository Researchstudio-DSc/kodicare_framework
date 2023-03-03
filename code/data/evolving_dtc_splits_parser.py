"""
class to parse the hdf5 of splits for cord 19 and robust test collection includes
- the generation of ids for dynamic test collection
- the generation of runs results
"""

import h5py
import numpy as np


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
