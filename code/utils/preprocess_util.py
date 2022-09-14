"""
contain functions for text preporcessing
"""
from langdetect import detect
from langdetect import DetectorFactory


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


def spacy_tokenizer(sentence, parser, stopwords, punctuations):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


def vectorize_text(text, vectorizer):
    return vectorizer.fit_transform(text)