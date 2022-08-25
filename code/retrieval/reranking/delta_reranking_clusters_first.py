from code.retrieval.reranking import delta_reranking_interface
from code.utils import io_util


def merge_reranked_docs(ordered_clusters, reranked_cluster_relevant_docs_map):
    reranked_relevant_docs = []
    for cluster_id in ordered_clusters:
        reranked_relevant_docs.extend(reranked_cluster_relevant_docs_map[cluster_id])
    return reranked_relevant_docs


class DeltaRerankingClustersFirst(delta_reranking_interface.DeltaRerankingInterface):
    def rerank_retrieval_output(self, base_retrieval_output_path, reranked_retrieval_output_path):
        reranked_retrieval_result = []
        base_retrieval_result = io_util.read_json(base_retrieval_output_path)
        for query_info in base_retrieval_result:
            ordered_clusters, cluster_relevant_docs_map, relevant_docs_scores_map = self.cluster_relevant_docs(
                query_info)
            reranked_cluster_relevant_docs_map = self.rerank_documents_per_cluster(cluster_relevant_docs_map,
                                                                                   relevant_docs_scores_map)
            reranked_relevant_docs = merge_reranked_docs(ordered_clusters, reranked_cluster_relevant_docs_map)
            delta_reranking_interface.add_reranked_query_result(query_info[delta_reranking_interface.MAP_KEY__QUERY],
                                                                query_info[delta_reranking_interface.MAP_KEY__QUERY_ID],
                                                                reranked_relevant_docs, relevant_docs_scores_map,
                                                                reranked_retrieval_result)

        io_util.write_json(reranked_retrieval_output_path, reranked_retrieval_result)

    def cluster_relevant_docs(self, query_info_map):
        ordered_clusters = []
        cluster_relevant_docs_map = {}
        relevant_docs_scores_map = {}
        for rel_doc in query_info_map[delta_reranking_interface.MAP_KEY__RELEVANT_DOCS]:
            cluster_id = self.doc_cluster_id_map[rel_doc[delta_reranking_interface.MAP_KEY__DOC_ID]]
            if cluster_id not in ordered_clusters:
                ordered_clusters.append(cluster_id)
                cluster_relevant_docs_map[cluster_id] = []
            cluster_relevant_docs_map[cluster_id].append(rel_doc[delta_reranking_interface.MAP_KEY__DOC_ID])
            relevant_docs_scores_map[rel_doc[delta_reranking_interface.MAP_KEY__DOC_ID]] = rel_doc[
                delta_reranking_interface.MAP_KEY__SCORE]
        return ordered_clusters, cluster_relevant_docs_map, relevant_docs_scores_map

    def rerank_documents_per_cluster(self, cluster_relevant_docs_map, relevant_docs_scores_map):
        reranked_cluster_relevant_docs_map = {}
        for cluster_id in cluster_relevant_docs_map:
            reranked_cluster_relevant_docs_map[cluster_id] = self.qprp_rerank(cluster_relevant_docs_map[cluster_id],
                                                                              relevant_docs_scores_map)
        return reranked_cluster_relevant_docs_map
