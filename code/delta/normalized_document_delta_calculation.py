from code.delta import delta_calculation_interface
from code.utils import io_util
from code.utils import preprocess_util
from code.utils import metrics_utils
from code.delta import clustering_interface

MAP_KEY__EMBEDDING = "embedding"


def vectorize_documents(df, vectorizer_model, processed_text_name):
    print("Vectorizing processed text ... ")
    text = df[processed_text_name].values
    vectors = preprocess_util.vectorize_text(text, vectorizer_model)
    return vectors.toarray()


def get_doc_id_text_vector_map(df, vectors, processed_text_name, doc_id_name):
    print("getting vectors map")
    doc_ids = df[doc_id_name].values
    processed_text = df[processed_text_name].values
    doc_id_text_vector_map = {}

    for (index, value) in enumerate(doc_ids):
        doc_id_text_vector_map[value] = {processed_text_name: processed_text[index], MAP_KEY__EMBEDDING: vectors[index]}

    return doc_id_text_vector_map


def get_clusters_documents_map(df, clusters_label, doc_id_label):
    clusters = {}
    for index, row in df.iterrows():
        if row[clusters_label] not in clusters:
            clusters[row[clusters_label]] = []
        clusters[row[clusters_label]].append(row[doc_id_label])
    return clusters


def calculate_distance_between_docs_pairs(clusters, doc_id_text_vector_map, similarity_method):
    print("Calculate distance between docs .. ")
    clusters_similarity_map = []
    for cluster_key in clusters:
        similarity_pairs = []
        for i in range(len(clusters[cluster_key]) - 1):
            doc_1 = doc_id_text_vector_map[clusters[cluster_key][i]][MAP_KEY__EMBEDDING]
            doc_2 = doc_id_text_vector_map[clusters[cluster_key][i + 1]][MAP_KEY__EMBEDDING]
            similarity_score = metrics_utils.calculate_similarity_between_vectors(similarity_method, doc_1, doc_2)
            similarity_pairs.append({
                delta_calculation_interface.MAP_KEY__DOC_ID_1: clusters[cluster_key][i],
                delta_calculation_interface.MAP_KEY__DOC_ID_2: clusters[cluster_key][i + 1],
                delta_calculation_interface.MAP_KEY__SIMILARITY_SCORE: similarity_score
            })
        clusters_similarity_map.append({
            delta_calculation_interface.MAP_KEY__CLUSTER_ID: cluster_key,
            delta_calculation_interface.MAP_KEY__SIMILARITY_PAIRS: similarity_pairs
        })
    return clusters_similarity_map


class NormalizedDocumentDeltaCalculation(delta_calculation_interface.DocumentDeltaCalculationInterface):
    def __init__(self, vectorizer_model, processed_text_name, similarity_method):
        self.vectorizer_model = vectorizer_model
        self.processed_text_name = processed_text_name
        self.similarity_method = similarity_method

    def calculate_document_delta_score(self, input_df_path, output_path):
        df = io_util.read_pickle(input_df_path)
        print(df.head)
        vectors = vectorize_documents(df, self.vectorizer_model, self.processed_text_name)

        # TODO change uid to be variable
        doc_id_text_vector_map = get_doc_id_text_vector_map(df, vectors, self.processed_text_name,
                                                            clustering_interface.MAP_KEY__UID)

        clusters = get_clusters_documents_map(df, clustering_interface.MAP_KEY__CLUSTER_LABEL,
                                              clustering_interface.MAP_KEY__UID)

        doc_similarity_pairs = calculate_distance_between_docs_pairs(clusters, doc_id_text_vector_map,
                                                                     self.similarity_method)

        io_util.write_json(output_path, doc_similarity_pairs)
