from code.retrieval.reranking import delta_reranking_interface
from code.utils import io_util


def add_reranked_query_result(query, query_id, reranked_docs, relevant_docs_scores_map, reranked_retrieval_result):
    relevant_docs_map = []
    for doc in reranked_docs:
        relevant_docs_map.append({
            delta_reranking_interface.MAP_KEY__DOC_ID: doc,
            delta_reranking_interface.MAP_KEY__SCORE: relevant_docs_scores_map[doc]
        })

    reranked_retrieval_result.append({
        delta_reranking_interface.MAP_KEY__QUERY: query,
        delta_reranking_interface.MAP_KEY__QUERY_ID: query_id,
        delta_reranking_interface.MAP_KEY__RELEVANT_DOCS: relevant_docs_map
    })


class DeltaRerankingClustersIgnored(delta_reranking_interface.DeltaRerankingInterface):
    def rerank_retrieval_output(self, base_retrieval_output_path, reranked_retrieval_output_path):
        base_retrieval_result = io_util.read_json(base_retrieval_output_path)
        reranked_retrieval_result = self.construct_reranked_retrieval_result(base_retrieval_result)

        io_util.write_json(reranked_retrieval_output_path, reranked_retrieval_result)

    def construct_reranked_retrieval_result(self, base_retrieval_result):
        reranked_retrieval_result = []
        for query in base_retrieval_result:
            relevant_docs_ids = []
            relevant_docs_scores_map = {}
            for rel_doc in query[delta_reranking_interface.MAP_KEY__RELEVANT_DOCS]:
                relevant_docs_ids.append(rel_doc[delta_reranking_interface.MAP_KEY__DOC_ID])
                relevant_docs_scores_map[rel_doc[delta_reranking_interface.MAP_KEY__DOC_ID]] = rel_doc[
                    delta_reranking_interface.MAP_KEY__SCORE]
            reranked_docs = self.qprp_rerank(relevant_docs_ids, relevant_docs_scores_map)
            add_reranked_query_result(query[delta_reranking_interface.MAP_KEY__QUERY],
                                     query[delta_reranking_interface.MAP_KEY__QUERY_ID],
                                     reranked_docs, relevant_docs_scores_map, reranked_retrieval_result)
        return reranked_retrieval_result
