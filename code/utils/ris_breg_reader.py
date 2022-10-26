import csv
from typing import Iterable, Union
from tqdm import tqdm

from code.utils.reader_util import CollectionReader
from code.indexing.index_util import Query


class RISBatchReader(CollectionReader):

    def __init__(self, collection_path, batch_size: int = 1024) -> None:
        super().__init__(collection_path)
        self.batch_size = batch_size
   

    def iterate(self) -> Union[Iterable, object]:
        # should either iterate through batches
        # or single documents
        if self.batch_size and self.batch_size > 0:
            for batch in self.iterate_batch():
                yield batch
        else:
            for doc in self.iterate_single():
                yield doc


    def iterate_batch(self):
        batch = []
        # go through all documents and return the documents as a batch of documents
        with open(self.collection_path, 'r') as fp:
            collection = csv.reader(fp, delimiter="\t", quotechar='"')
            for document in tqdm(collection):
                index_document = self.read(document)
                batch.append(index_document)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        # finally yield the last documents
        if len(batch) > 0:
            yield batch


    def iterate_single(self):
        # go through all documents and return the documents as a batch of documents
        with open(self.collection_path, 'r') as fp:
            collection = csv.reader(fp, delimiter="\t", quotechar='"')
            for document in tqdm(collection):
                index_document = self.read(document)
                yield index_document


class RisBregReader(RISBatchReader):

    def __init__(self, collection_path, batch_size: int = 1024) -> None:
        super().__init__(collection_path, batch_size)


    def read(self, document):
        p_id, passage_text = document
        document_obj = {
            "passage_id": p_id,
            "passage_text": passage_text
        }
        return (document_obj["passage_id"],document_obj)
    

    def to_string(self, document_obj):
        return f'{document_obj["passage_id"]}'


class ESQueryReader:

    def __init__(self, queries_path) -> None:
        self.queries_path = queries_path
    

    def read(self):
        with open(self.collection_path, 'r') as fp:
            queries = csv.reader(fp, delimiter="\t", quotechar='"')
        queries = []
        for q_id, q_text in queries:
            queries.append(Query(
                id=q_id,
                data=q_text
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
                    "fields": ["passage_text"]
                }
            }
        }
        return data