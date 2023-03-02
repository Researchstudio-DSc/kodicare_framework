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
    return [tokens.count(w) for w in vocab]


class BOWRepresentation(text_representation_interface.TextRepresentationInterface):
    def __init__(self, vocab, lang='en'):
        super().__init__(lang=lang)
        self.vocab = vocab

    def represent_text(self, text):
        tokens = preprocess_util.execute_common_preprocess_pipeline(
            text, string.punctuation, language=preprocess_util.LANGUAGE_CODE_LANGUAGE__MAP[self.lang])

        vectors = bow_vectorize(tokens, self.vocab)

        return vectors
