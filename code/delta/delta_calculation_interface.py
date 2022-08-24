MAP_KEY__DOC_ID_1 = "doc_id_1"
MAP_KEY__DOC_ID_2 = "doc_id_2"
MAP_KEY__SIMILARITY_SCORE = "similarity_score"
MAP_KEY__SIMILARITY_PAIRS = "similarity_pairs"
MAP_KEY__CLUSTER_ID = "cluster_id"


class DocumentDeltaCalculationInterface:

    def calculate_document_delta_score(self, input_df_path, output_path):
        """
        a function to calculate a numerical score between pairs of document in clusters
        :param input_df_path: the path of the data frame that stores documents and clusters information
        :param output_path: the output path for the calculated pairs scores
        :return:
        """
        pass
