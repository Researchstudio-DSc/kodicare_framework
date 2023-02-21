"""
implement the text representation class for bag of words representation
"""
import string

from code.representations import text_representation_interface
from code.utils import preprocess_util


def bow_vectorize(tokens, vocab):
    '''
    This function takes list of words in a sentence as input
    and returns a vector of size of filtered_vocab.It puts 0 if the
    word is not present in tokens and count of token if present.
    '''
    vector = []
    for w in vocab:
        vector.append(tokens.count(w))
    return vector


class BOWRepresentation(text_representation_interface.TextRepresentationInterface):
    def represent_text(self, text):
        # step 1: tokenize the text
        stopwords = preprocess_util.get_stopwords(language=self.LANGUAGE_CODE_LANGUAGE__MAP[self.lang])

        tokens = preprocess_util.word_tokenize(text)
        tokens = preprocess_util.remove_stopwords(tokens, stopwords)
        tokens = preprocess_util.remove_punctuation(tokens, string.punctuation)

        # TODO: update the vocab with the new text .. we want to have access to previous vocab somehow
        vocab = list(set(tokens))

        vectors = bow_vectorize(tokens, vocab)

        return vocab, vectors
