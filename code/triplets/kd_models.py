from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFDelta:

    def __init__(self, max_df=0.75, min_df=10) -> None:
        self.max_df = max_df
        self.min_df = min_df
        self.tfidf_vect = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df)
    

    def create_embeddings(self, document_iter):
        X = self.tfidf_vect.fit_transform(document_iter)
        return X