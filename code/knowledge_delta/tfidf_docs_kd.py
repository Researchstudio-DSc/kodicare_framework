"""
class that implements kd between document collections using TFIDF representation
"""

import numpy as np

from code.knowledge_delta import docs_knowledge_delta_interface
from code.utils import io_util


def min_max_np_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def construct_tc_rep(vocab_dict, doc_vectors):
    tc_rep = np.zeros((len(doc_vectors), len(vocab_dict)))
    for doc_ind, doc_vec in enumerate(doc_vectors):
        for k, v in doc_vec:
            tc_rep[doc_ind][k] = v

    return tc_rep


def calculate_box_plot_boundaries(data):
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    uw = q3 + 1.5 * (q3 - q1)
    lw = q3 - 1.5 * (q3 - q1)
    return lw, q1, q2, q3, uw


class TIFIDFDocsKD(docs_knowledge_delta_interface.DocsKnowledgeDeltaInterface):
    def __init__(self, rep_dir, vocab_dict, epochs, start_step, end_step, generate_stats=False):
        self.vocab_dict = vocab_dict
        self.rep_dir = rep_dir
        self.epochs = epochs
        self.start_step = start_step
        self.end_step = end_step
        if generate_stats:
            self.generate_vocab_stats()

    # generate summary of the vocab occurrences using the box plot
    def generate_vocab_stats(self):
        for tc_ind in range(0, self.epochs):
            vec = io_util.read_pickle(io_util.join(self.rep_dir, str(tc_ind) + '_tfidf_vec.pkl'))
            data = construct_tc_rep(self.vocab_dict, vec)
            normalized_data = min_max_np_normalize(data)
            lw, q1, q2, q3, uw = calculate_box_plot_boundaries(normalized_data[np.nonzero(normalized_data)])

            print(tc_ind, '-', 'Box plot boundries', lw, q1, q2, q3, uw)

            # vocab statistics
            vocab_stat = np.zeros((3, len(self.vocab_dict)))
            for voc_ind in range(len(self.vocab_dict)):
                vocab_rep = normalized_data[:, voc_ind]
                vocab_stat[0][voc_ind] = ((lw < vocab_rep) & (vocab_rep < q1)).sum()
                vocab_stat[1][voc_ind] = ((q1 < vocab_rep) & (vocab_rep < q3)).sum()
                vocab_stat[2][voc_ind] = ((q3 < vocab_rep) & (vocab_rep < uw)).sum()
            io_util.write_pickle(vocab_stat, io_util.join(self.rep_dir, str(tc_ind) + '_vocab_stats.pkl'))

    def calculate_kd(self):
        diff_vectors_l = []  # lower range
        diff_vectors_m = []  # middle range
        diff_vectors_u = []  # upper range
        for step in range(self.start_step, self.end_step):
            for tc_ind in range(0, self.epochs - step):
                for tc_ind2 in range(tc_ind + step, self.epochs - step):
                    vocab_stat_1 = io_util.read_pickle(io_util.join(self.rep_dir, str(tc_ind) + '_vocab_stats.pkl'))
                    vocab_stat_2 = io_util.read_pickle(io_util.join(self.rep_dir, str(tc_ind2) + '_vocab_stats.pkl'))
                    diff_vectors_l.append(
                        np.subtract(min_max_np_normalize(vocab_stat_1[0, :]), min_max_np_normalize(vocab_stat_2[0, :])))
                    diff_vectors_m.append(
                        np.subtract(min_max_np_normalize(vocab_stat_1[1, :]), min_max_np_normalize(vocab_stat_2[1, :])))
                    diff_vectors_u.append(
                        np.subtract(min_max_np_normalize(vocab_stat_1[2, :]), min_max_np_normalize(vocab_stat_2[2, :])))
        return diff_vectors_l, diff_vectors_m, diff_vectors_u
