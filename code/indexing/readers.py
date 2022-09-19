import json
import os
from tqdm import tqdm

class BatchReader:

    def __init__(self, batch_size: int = 1024) -> None:
        super().__init__()
        self.batch_size = batch_size
    

    def read(self, document_data):
        raise NotImplementedError
    

    def iterate(self, folder):
        batch = []
        # go through all documents and return the documents as a batch of documents
        for fname in tqdm(os.listdir(folder)):
            with open(os.path.join(folder, fname), 'r') as fp:
                document_data = json.load(fp)
            # read might return multiple documents, e.g. when they are split into paragraphs
            for index_document in self.read(document_data):
                batch.append(index_document)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        # finally yield the last documents
        if len(batch) > 0:
            yield batch


class CORD19Reader(BatchReader):

    def __init__(self, batch_size: int = 64) -> None:
        super().__init__(batch_size)


    def read(self, document_data):
        abstract = " ".join([p["text"] for p in document_data["paragraphs"] if p["section"]["text"] == "Abstract"])
        document_obj = {
            "uid": document_data["uid"],
            "doc_id": document_data["doc_id"],
            "title": document_data["metadata"]["title"],
            "abstract": abstract
        }
        return [(document_obj["uid"],document_obj)]
    

    def build(self, query, size):
        data = {
            "from" : 0, 
            "size" : size,
            "query": {
                "multi_match" : {
                    "query": query,
                    "type": "cross_fields",
                    "fields": ["title", "abstract"]
                }
            }
        }
        return data
    

    def to_string(self, document_obj):
        return f"{document_obj['title']}\n{document_obj['abstract']}"


class CORD19ParagraphReader(BatchReader):

    def __init__(self, batch_size: int = 64) -> None:
        super().__init__(batch_size)


    def read(self, document_data):
        paragraph_objs = []
        for idx, p in enumerate(document_data["paragraphs"]):
            document_obj = {
                "uid": document_data["uid"],
                "doc_id": document_data["doc_id"],
                "paragraph_id": f'{document_data["uid"]}_{idx}',
                "paragraph_number": idx,
                "title": document_data["metadata"]["title"],
                "section": p["section"]["text"],
                "paragraph_text": p["text"]
            }
            paragraph_objs.append((document_obj["paragraph_id"], document_obj))
        return paragraph_objs
    

    def build(self, query, size):
        data = {
            "from" : 0, 
            "size" : size,
            "query": {
                "multi_match" : {
                    "query": query,
                    "type": "cross_fields",
                    "fields": ["title", "section", "paragraph_text"]
                }
            }
        }
        return data
    

    def to_string(self, document_obj):
        return f"{document_obj['title']}\n{document_obj['section']}\n{document_obj['paragraph_text']}" 