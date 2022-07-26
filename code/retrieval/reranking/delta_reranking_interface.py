from code.utils import io_util
from code.delta import clustering_interface
from code.delta import delta_calculation_interface
import sys
import math

MAP_KEY__QUERY = "query"
MAP_KEY__QUERY_ID = "query_id"
MAP_KEY__RELEVANT_DOCS = "relevant_docs"
MAP_KEY__DOC_ID = "doc_id"
MAP_KEY__SCORE = "score"


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


def add_reranked_query_result(query, query_id, reranked_docs, relevant_docs_scores_map, reranked_retrieval_result):
    relevant_docs_map = []
    for doc in reranked_docs:
        relevant_docs_map.append({
            MAP_KEY__DOC_ID: doc,
            MAP_KEY__SCORE: relevant_docs_scores_map[doc]
        })

    reranked_retrieval_result.append({
        MAP_KEY__QUERY: query,
        MAP_KEY__QUERY_ID: query_id,
        MAP_KEY__RELEVANT_DOCS: relevant_docs_map
    })


class DeltaRerankingInterface:
    def __init__(self, doc_clusters_path, doc_pairs_similarity_path):
        """
        init class of the reranking
        :param doc_clusters_path: json path of documents clusters
        """
        self.doc_cluster_id_map = init_doc_cluster_id_map(doc_clusters_path)
        self.doc_pairs_similarity_map = init_doc_pairs_similarity_map(doc_pairs_similarity_path)

    def qprp_rerank(self, relevant_docs, relevant_docs_scores_map):
        """
        reranking a list of relevant docs according to the QPRP
        :param relevant_docs: list of relevant docs to the query
        :param relevant_docs_scores_map: map of relevant docs and their relevant probability to the query
        :return: reranked docs according to the QPRP
        """

        relevant_docs_set = set(relevant_docs[1:])
        reranked_docs = []
        reranked_docs.append(relevant_docs[0])
        while len(relevant_docs_set) != 0:
            next_doc_uid = ""
            next_doc_score = -1000

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
        return reranked_docs

    def rerank_retrieval_output(self, base_retrieval_output_path, reranked_retrieval_output_path):
        """
        function to rerank the retrieval output for the list of queries
        :param base_retrieval_output_path: the json file of the results from the execution of the base retrieval method
        :param reranked_retrieval_output_path: the output path to save the reranked output
        :return:
        """
        pass
