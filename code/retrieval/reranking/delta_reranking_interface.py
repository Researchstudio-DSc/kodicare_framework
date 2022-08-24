from code.utils import io_util
from code.delta import clustering_interface
from code.delta import delta_calculation_interface
import sys
import math


def init_doc_cluster_id_map(doc_clusters_path):
    df = io_util.read_pickle(doc_clusters_path)
    doc_cluster_id_map = {}
    for index, row in df.iterrows():
        doc_cluster_id_map[row[clustering_interface.MAP_KEY__UID]] = row[clustering_interface.MAP_KEY__CLUSTER_LABEL]
    return doc_cluster_id_map


def init_doc_pairs_similarity_map(doc_pairs_similarity_path):
    clusters_info = io_util.read_json(doc_pairs_similarity_path)
    doc_pairs_similarity_map = {}

    for cluster in clusters_info:
        for sim_pair in cluster[delta_calculation_interface.MAP_KEY__SIMILARITY_PAIRS]:
            key = sim_pair[delta_calculation_interface.MAP_KEY__DOC_ID_1] + "/" + sim_pair[
                delta_calculation_interface.MAP_KEY__DOC_ID_2]
            doc_pairs_similarity_map[key] = sim_pair[delta_calculation_interface.MAP_KEY__SIMILARITY_SCORE]
    return doc_pairs_similarity_map


class DeltaRerankingInterface:
    def __init__(self, base_retrieval_output_path, doc_clusters_path, doc_pairs_similarity_path):
        """
        init class of the reranking
        :param base_retrieval_output_path: the json file of the results from the execution of the base retrieval method
        :param doc_clusters_path: json path of documents clusters
        """
        self.doc_cluster_id_map = init_doc_cluster_id_map(doc_clusters_path)
        self.doc_pairs_similarity_map = init_doc_pairs_similarity_map(doc_pairs_similarity_path)
        self.base_retrieval_output_path = base_retrieval_output_path

    def qprp_rerank(self, relevant_docs, relevant_docs_scores_map):
        """
        reranking a list of relevant docs according to the QPRP
        :param relevant_docs: list of relevant docs to the query
        :param relevant_docs_scores_map: map of relevant docs and their relevant probability to the query
        :return: reranked docs according to the QPRP
        """

        relevant_docs_set = set(relevant_docs[1:])
        reranked_docs = []
        reranked_docs[0] = relevant_docs[0]
        while len(relevant_docs_set) != 0:
            next_doc_uid = ""
            next_doc_score = -sys.maxint - 1

            for doc in relevant_docs_set:
                current_score = relevant_docs_scores_map[doc]
                for reranked_doc in reranked_docs:
                    correlation = 0
                    if doc + "/" + reranked_doc in self.doc_pairs_similarity_map:
                        correlation = self.doc_pairs_similarity_map[doc + "/" + reranked_doc]
                    elif reranked_doc + "/" + doc in self.doc_pairs_similarity_map:
                        correlation = self.doc_pairs_similarity_map[reranked_doc + "/" + doc]
                    current_score -= (math.sqrt(relevant_docs_scores_map[doc]) * math.sqrt(
                        relevant_docs_scores_map[reranked_doc]) * correlation)
                if next_doc_score < current_score:
                    next_doc_score = current_score
                    next_doc_uid = doc

            relevant_docs_set.remove(next_doc_uid)
            reranked_docs.append(next_doc_uid)
