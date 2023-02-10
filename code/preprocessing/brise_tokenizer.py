import spacy
import re


class RisBregTokenizer:

    def __init__(self, spacy_model_name) -> None:
        self.spacy_model_name = spacy_model_name
        self.nlp = spacy.load(spacy_model_name)
        self.identify_single_p = re.compile(r'((§\s?(?P<p>\d+(\s?[a-z])?)\b)|(Art\s?(?P<art>\d+(\s?[a-z])?)\b))(\sAbs\.?\s?(?P<abs>\d+(\s?[a-z])?\b))?(\slit\.?\s?(?P<lit>[a-z])\b)?(\sZ\.? ?(?P<zahl>(\d+(\s?[a-z])?\b|[a-z]\b)))?(?![.);-])')

    def is_excluded(self, token):
        criteria = [
            token.is_punct,
            token.is_digit,
            token.is_stop,
            token.like_num
        ]
        return any(criteria)
    
    def get_lemma_tokens(self, doc):
        return [self.process_token(t.lemma_) for t in doc if not self.is_excluded(t)]
    

    def process_token(self, token: str):
        # normalize law
        token = self.normalize_norm(token)
        # lower capitlized words but keep acronyms
        uppercase_letters = [c for c in token if c.isupper()]
        if token[0].isupper() and len(uppercase_letters) == 1:
            token = token.lower()
        return token.replace(" ", "_")
    

    def normalize_norm(self, token):
        match = self.identify_single_p.search(token)
        if match:
            ids = [
                match.group('p'),
                match.group('art'),
                match.group('abs'),
                match.group('lit'),
                match.group('zahl')
            ]
            signs = ["§", "Art", "Abs", "lit", "Z"]
            statute_id = [f"{sign} {s}" for s, sign in zip(ids, signs) if s != None]
            return " ".join(statute_id)
        else:
            return token
    

    def batch_tokenize(self, batch):
        doc_tokens = []
        for doc in self.nlp.pipe(batch):
            doc_tokens.append(self.process_doc_tokens(doc))
        return doc_tokens
    

    def process_doc_tokens(self, doc):
        matches = list(self.identify_single_p.finditer(doc.text))
        with doc.retokenize() as retokenizer:
            for match in matches:
                attrs = {"LEMMA": match.group(0)}
                s = match.start()
                e = match.end()
                spacy_span = doc.char_span(s, e, label="LAW")
                # if it is not an exact match with the tokens, ignore it
                if spacy_span == None:
                    continue
                retokenizer.merge(spacy_span, attrs=attrs)
        return self.get_lemma_tokens(doc)
    

    def tokenize(self, text):
        doc = self.nlp(text)
        return self.process_doc_tokens(doc)