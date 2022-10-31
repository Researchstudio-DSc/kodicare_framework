import spacy


class RisBregTokenizer:

    def __init__(self, spacy_model_name) -> None:
        self.spacy_model_name = spacy_model_name
        self.nlp = spacy.load(spacy_model_name)

    
    def get_lemma_tokens(self, doc):
        return [t.lemma_ for t in doc]
    

    def batch_tokenize(self, batch):
        doc_tokens = []
        for doc in self.nlp.pipe(batch):
            doc_tokens.append(self.get_lemma_tokens(doc))
        return doc_tokens