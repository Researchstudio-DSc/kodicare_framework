"""
An interface to be implemented per representation to construct training data for KD~RD prediction
"""
class TrainingDataBuilderInterface:
    def construct_training_data(self):
        """
        function to construct the training data including the feature vector and the class labels
        :return: df of features and class label
        """

    def build_kd_feature_vector(self):
        """
        function to construct the feature vector of knowledge delta
        :return: df of features
        """

    def build_rd_df(self):
        """
        function to build a df of result delta all combination of existing systems and evaluation measure
        we can use any of them as class label for training.
        :return: df of result delta
        """