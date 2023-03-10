"""
contain functions for text preporcessing
"""
from langdetect import DetectorFactory
from langdetect import detect
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

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


def remove_short_tokens(tokens, size=3):
    return [word for word in tokens if len(word) > size]


def lemmatize_tokens(tokens, language='english'):
    if language != 'english':
        return tokens
    return [WordNetLemmatizer().lemmatize(token, pos='v') for token in tokens]


def stem_tokens(tokens, language='english'):
    stemmer = SnowballStemmer(language)
    return [stemmer.stem(token) for token in tokens]


def spacy_tokenizer(sentence, parser, stopwords, punctuations):
    tokens = parser(sentence)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    tokens = [word for word in tokens if word not in stopwords and word not in punctuations]
    tokens = " ".join([i for i in tokens])
    return tokens


def vectorize_text(text, vectorizer):
    return vectorizer.fit_transform(text)


def execute_common_preprocess_pipeline(text, punctuation, language='english'):
    stopwords = get_stopwords(language=language)

    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens, stopwords)
    tokens = remove_punctuation(tokens, punctuation)
    tokens = remove_short_tokens(tokens,)
    tokens = lemmatize_tokens(tokens, language=language)
    tokens = stem_tokens(tokens, language=language)

    return tokens
