"""
contain functions for text preporcessing
"""
from langdetect import DetectorFactory
from langdetect import detect
from nltk import tokenize
from nltk.corpus import stopwords

LANGUAGE_CODE_LANGUAGE__MAP = {'en': 'english', 'fr': 'french', 'de': 'german'}


def detect_text_language(text):
    # set seed
    DetectorFactory.seed = 0

    lang = "en"
    try:
        if len(text) > 50:
            lang = detect(" ".join(text[:50]))
        elif len(text) > 0:
            lang = detect(" ".join(text[:len(text)]))
    except Exception as e:
        all_words = set(text)
        try:
            lang = detect(" ".join(all_words))
        except Exception as e:
            lang = ""
            pass
    return lang


def get_stopwords(language='english'):
    return set(stopwords.words(language))


def word_tokenize(text):
    return [word.lower() for word in tokenize.word_tokenize(text)]


def remove_stopwords(tokens, stopwords):
    return [word for word in tokens if word not in stopwords]


def remove_punctuation(tokens, punctuations):
    return [word for word in tokens if word not in punctuations]


def spacy_tokenizer(sentence, parser, stopwords, punctuations):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


def vectorize_text(text, vectorizer):
    return vectorizer.fit_transform(text)
