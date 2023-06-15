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

        # sum the vectors
        bow_dict = {}
        tfidf_dict = {}
        for k, v in self.vocab_dict.iteritems():
            bow_dict[k] = tfidf_dict[k] = 0

        for vec in bow_vectors:
            for name, num in vec:
                bow_dict[name] += num
        for vec in tfidf_vectors:
            for name, num in vec:
                tfidf_dict[name] += num

        # using map
        merged_bow_vec = list(map(tuple, bow_dict.items()))
        merged_tfidf_vec = list(map(tuple, tfidf_dict.items()))
        return merged_bow_vec, merged_tfidf_vec
