import pandas as pd
from gensim import models

from code.representations import document_collection_representation_interface


class TFIDFDocCollectionRepresentation(
    document_collection_representation_interface.DocumentCollectionRepresentationInterface):
    def __init__(self, vocab_dict, docs_collection_path, lang='en'):
        super().__init__(lang=lang)
        self.vocab_dict = vocab_dict
        self.docs_collection_path = docs_collection_path

    def bow_vectorize(self, tokens):
        '''
        This function takes list of words in a text as input, vocab dictionary from gensim
        and returns the frequency of the words in the text
        '''
        return self.vocab_dict.doc2bow(tokens)

    def represent_document_collection(self):
        df = pd.read_json(self.docs_collection_path)
        df['merged_text'] = df['title'] + ' ' + df['contents']

        bow_vectors = [self.bow_vectorize(doc) for doc in df['merged_text']]
        print(len(bow_vectors))
        tfidf = models.TfidfModel(bow_vectors)
        tfidf_vectors = tfidf[bow_vectors]

        return bow_vectors, tfidf_vectors
