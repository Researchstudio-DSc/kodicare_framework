MAP_KEY__UID = "uid"
MAP_KEY__ABSTRACT = "abstract"
MAP_KEY__BODY_TEXT = "body_text"
MAP_KEY__TITLE = "title"
MAP_KEY__ABSTRACT_SUMMARY = "abstract_summary"
MAP_KEY__ABSTRACT_WORD_COUNT = "abstract_word_count"
MAP_KEY__BODY_WORD_COUNT = "body_word_count"
MAP_KEY__BODY_UNIQUE_WORD_COUNT = "body_unique_word_count"
MAP_KEY__LANGUAGE = "language"
MAP_KEY__PROCESSED_TEXT = "processed_text"
MAP_KEY__CLUSTER_LABEL = "cluster_label"


class ClusteringInterface:

    def build_clusters(self, input_path, output_path):
        """
        building clusters of a set of document based on their similarity
        :param input_path: the input directory of the documents cluster set
        :param output_path: the output path of the generated clusters
        :return:
        """
        pass
