from code.delta import delta_calculation_interface
from code.utils import io_util
from code.utils import preprocess_util

MAP_KEY__EMBEDDING = "embedding"


def vectorize_documents(df, vectorizer_model, processed_text_name):
    print("Vectorizing processed text ... ")
    text = df[processed_text_name].values
    vectors = preprocess_util.vectorize_text(text, vectorizer_model)
    return vectors.toarray()


def get_doc_id_text_vector_map(df, vectors, processed_text_name, doc_id_name):
    doc_ids = df[doc_id_name].values
    processed_text = df[processed_text_name].values
    doc_id_text_vector_map = {}

    for (index, value) in enumerate(doc_ids):
        doc_id_text_vector_map[value] = {processed_text_name: processed_text[index], MAP_KEY__EMBEDDING: vectors[index]}

    return doc_id_text_vector_map


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
        doc_id_text_vector_map = get_doc_id_text_vector_map(df, vectors, self.processed_text_name, 'uid')
