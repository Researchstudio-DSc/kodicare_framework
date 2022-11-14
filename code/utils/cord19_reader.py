import json
import os
from tqdm import tqdm
from lxml import etree
from code.indexing.index_util import Query

from hydra.utils import instantiate

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
            "cord_uid": document_data["cord_uid"],
            "title": document_data["metadata"]["title"],
            "abstract": abstract
        }
        return [(document_obj["uid"],document_obj)]
    

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
                "cord_uid": document_data["cord_uid"],
                "paragraph_id": f'{document_data["uid"]}_{idx}',
                "paragraph_number": idx,
                "title": document_data["metadata"]["title"],
                "section": p["section"]["text"],
                "paragraph_text": p["text"]
            }
            paragraph_objs.append((document_obj["paragraph_id"], document_obj))
        return paragraph_objs
    

    def to_string(self, document_obj):
        return f"{document_obj['title']}\n{document_obj['section']}\n{document_obj['paragraph_text']}"


class ESQueryReader:

    def __init__(self, queries_path, index_fields) -> None:
        self.queries_path = queries_path
        self.index_fields = index_fields
    

    def read(self):
        with open(self.queries_path, 'r') as fp:
            topics = etree.parse(fp).getroot()
        queries = []
        for topic in topics:
            query = topic[0]
            queries.append(Query(
                id=topic.attrib['number'],
                data=query.text
            ))
        return queries
    

    def build(self, query, size):
        data = {
            "from" : 0, 
            "size" : size,
            "query": {
                "multi_match" : {
                    "query": query,
                    "type": "cross_fields",
                    "fields": [f for f in self.index_fields]
                }
            }
        }
        return data


class FAISSQueryReader:

    def __init__(self, queries_path, model_cfg) -> None:
        self.queries_path = queries_path
        self.model = instantiate(model_cfg)
    

    def read(self):
        with open(self.queries_path, 'r') as fp:
            topics = etree.parse(fp).getroot()

        queries = []
        for idx, topic in enumerate(topics):
            query = topic[0]
            question = topic[1]
            assert query.tag == "query"
            assert question.tag == "question"
            query_vector = self.model.get_vector(f"{query.text} {question.text}")
            queries.append(Query(
                id=idx,
                data=query_vector
            ))
