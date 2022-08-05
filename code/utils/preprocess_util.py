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
