from code.utils import io_util
from code.delta import clustering_interface
from code.delta import delta_calculation_interface


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
            key = sim_pair[delta_calculation_interface.MAP_KEY__DOC_ID_1] + "/" + sim_pair[delta_calculation_interface.MAP_KEY__DOC_ID_2]
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

