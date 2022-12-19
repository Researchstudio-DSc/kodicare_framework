import csv
import json
import os
from typing import Iterable, Union
from tqdm import tqdm
from code.utils.metrics_utils import kli_divergence, get_token_counts
from code.preprocessing.brise_tokenizer import RisBregTokenizer

from code.utils.reader_util import CollectionReader
from code.indexing.index_util import Query


class RISBatchReader(CollectionReader):

    def __init__(self, data_dir, collection, batch_size: int = 1024) -> None:
        super().__init__(data_dir, collection)
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

    def __init__(self, data_dir, collection, batch_size: int = 1024) -> None:
        super().__init__(data_dir, collection, batch_size)


    def read(self, document):
        passage_id, passage_text, gz = document
        document_obj = {
            "document_id": passage_id,
            "passage_text": passage_text,
            "gz": gz
        }
        return (document_obj["document_id"],document_obj)
    

    def to_string(self, document_obj):
        return f'{document_obj["document_id"]}'


class ESQueryReader:

    def __init__(self, queries=None, data_dir=None, collection_prob_dict_file=None, kli_ratio=0.6, filter=True) -> None:
        self.queries_path = os.path.join(data_dir, queries)
        with open(os.path.join(data_dir, collection_prob_dict_file), "r") as fp:
            self.collection_prob_dict = json.load(fp)
        self.kli_ratio = kli_ratio
        self.filter = filter
        self.tokenizer = RisBregTokenizer("de_core_news_md")
    

    def read(self):
        with open(self.queries_path, 'r') as fp:
            queries_tsv = csv.reader(fp, delimiter="\t", quotechar='"')
            queries = []
            for q_id, q_text, target_gz in queries_tsv:
                q_tokens = self.tokenizer.tokenize(q_text)
                q_token_counts = get_token_counts(q_tokens)
                q_token_probs = kli_divergence(
                    terms=set(q_tokens),
                    document_counts=q_token_counts, 
                    collection_prob_dict=self.collection_prob_dict)
                
                select_tokens_count = int(len(q_token_probs)*self.kli_ratio)
                q_token_probs = q_token_probs[:select_tokens_count]
                queries.append(Query(
                    id=q_id,
                    data=(" ".join([t for t, p in q_token_probs]), target_gz)
                ))
        return queries
    

    def build(self, query, size):
        query_text, target_gz = query
        data = {
            "from" : 0, 
            "size" : size,
            "query": {
                "bool": {
                    "must":[{
                        "multi_match" : {
                            "query": query_text,
                            "type": "cross_fields",
                            "fields": ["passage_text"]
                        }
                    }]
                }
                
            }
        }
        if self.filter:
            data["query"]["bool"]["filter"] = [{
                    "term": { "gz": target_gz}
                }]
        return data