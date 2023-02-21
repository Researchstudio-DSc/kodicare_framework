class TextRepresentationInterface:
    LANGUAGE_CODE_LANGUAGE__MAP = {'en': 'english', 'fr': 'french', 'de': 'german'}

    def __init__(self, lang='en'):
        """
        :param lang: the language code of the text default 'en'
        """
        self.lang = lang

    def represent_text(self, text):
        """
        function that return representation of text (e.g. vector representation)
        implemented differently according to the type of representation
        :param text: the text to be represented
        :return:
        """
        return
