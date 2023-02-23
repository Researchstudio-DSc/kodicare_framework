"""
implement the text representation class for TF-IDF of words representation
"""

import numpy as np
import string

from code.representations import text_representation_interface
from code.utils import preprocess_util


def calculate_tfidf(tokens, df, n_docs):
    tokens_count_map = count_term_occurrence(tokens)
    tf_idf = {}
    for token in np.unique(tokens):
        tf = tokens_count_map[token] / len(tokens)
        idf = np.log(n_docs / (df[token] + 1))
        tf_idf[token] = tf * idf
    return tf_idf


def count_term_occurrence(tokens):
    tokens_count_map = {}
    for token in tokens:
        if token not in tokens_count_map:
            tokens_count_map[token] = 0
        tokens_count_map[token] += 1
    return tokens_count_map


def convert_tfidf_to_vector(tf_idf, vocab):
    tfidf_vector = np.zeros(len(vocab))
    for (index, word) in enumerate(vocab):
        if word in tf_idf:
            tfidf_vector[index] = tf_idf[word]
    return tfidf_vector


class TFIDFRepresentation(text_representation_interface.TextRepresentationInterface):
    def __init__(self, vocab, df, n_docs, lang='en'):
        super().__init__(lang=lang)
        self.vocab = vocab
        self.df = df
        self.n_docs = n_docs

    def represent_text(self, text):
        stopwords = preprocess_util.get_stopwords(language=preprocess_util.LANGUAGE_CODE_LANGUAGE__MAP[self.lang])

        tokens = preprocess_util.word_tokenize(text)
        tokens = preprocess_util.remove_stopwords(tokens, stopwords)
        tokens = preprocess_util.remove_punctuation(tokens, string.punctuation)

        tfidf = calculate_tfidf(tokens, self.df, self.n_docs)
        return convert_tfidf_to_vector(tfidf, self.vocab)
