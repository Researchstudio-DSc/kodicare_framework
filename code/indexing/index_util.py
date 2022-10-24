

class Query:
    KEY_SCORE="score"
    KEY_SOURCE="source"

    def __init__(self, id, data, relevant_docs=None) -> None:
        self.id = id
        self.data = data
        self.relevant_docs = relevant_docs