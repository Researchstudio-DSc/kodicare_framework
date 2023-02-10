import csv
import json
import os
from typing import Iterable, Union
from tqdm import tqdm
from code.utils.metrics_utils import kli_divergence, get_token_counts, idf_scores, plm
from code.preprocessing.brise_tokenizer import RisBregTokenizer

from code.utils.reader_util import CollectionReader
from code.indexing.index_util import Query


class RISBatchReader(CollectionReader):

    def __init__(self, data_dir, collection, batch_size: int = 1024) -> None:
        super().__init__(data_dir, collection)
        self.batch_size = batch_size
        self.tokenizer = RisBregTokenizer("de_core_news_md")
   

    def iterate(self) -> Union[Iterable, object]:
        # should either iterate through batches
        # or single documents
        if self.batch_size and self.batch_size > 0:
            for batch in self.iterate_batch():
                yield batch
        else:
            for doc in self.iterate_single():
                yield doc
    

    def tokenize_batch(self, batch):
        batch_texts = [doc["passage_text"] for doc_id, doc in batch]
        doc_tokens = self.tokenizer.batch_tokenize(batch_texts)
        for (doc_id, doc), doc_t in zip(batch, doc_tokens):
            doc["passage_text_tokenized"] = " ".join(doc_t)
        return batch


    def iterate_batch(self):
        batch = []
        # go through all documents and return the documents as a batch of documents
        with open(self.collection_path, 'r', encoding='utf-8') as fp:
            collection = csv.reader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
            for document in tqdm(collection):
                index_document = self.read(document)
                batch.append(index_document)
                if len(batch) == self.batch_size:
                    yield self.tokenize_batch(batch)
                    batch = []
        # finally yield the last documents
        if len(batch) > 0:
            yield self.tokenize_batch(batch)


    def iterate_single(self):
        # go through all documents and return the documents as a batch of documents
        with open(self.collection_path, 'r') as fp:
            collection = csv.reader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
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

    def __init__(self, queries=None, data_dir=None, collection_prob_dict_file=None, kw_extractor="kli", query_term_ratio=0.6, smooth_factor=0.5, filter=True) -> None:
        self.queries_path = os.path.join(data_dir, queries)
        with open(os.path.join(data_dir, collection_prob_dict_file), "r") as fp:
            self.collection_prob_dict = json.load(fp)
        self.kw_extractor = kw_extractor
        self.query_term_ratio = query_term_ratio
        self.filter = filter
        self.smooth_factor = smooth_factor
        self.tokenizer = RisBregTokenizer("de_core_news_md")
    

    def read(self):
        with open(self.queries_path, 'r') as fp:
            queries_tsv = csv.reader(fp, delimiter="\t", quotechar='"')
            queries = []
            for q_id, q_text, target_gz in queries_tsv:
                if self.query_term_ratio and self.kw_extractor:
                    q_tokens = self.tokenizer.tokenize(q_text)
                    #print(q_tokens)
                    q_token_counts = get_token_counts(q_tokens)
                    if self.kw_extractor == "kli":
                        q_token_scores = kli_divergence(
                            terms=set(q_tokens),
                            document_term_counts=q_token_counts, 
                            collection_prob_dict=self.collection_prob_dict)
                    elif self.kw_extractor == "idf":
                        q_token_scores = idf_scores(
                            terms=set(q_tokens),
                            idf_dict=self.collection_prob_dict
                        )
                    elif self.kw_extractor == "plm":
                        q_token_scores = plm(
                            terms=q_tokens,
                            document_term_counts=q_token_counts,
                            collection_prob_dict=self.collection_prob_dict,
                            smooth_factor=self.smooth_factor,
                            max_steps=50
                        )
                    else:
                        assert False
                    #print(q_token_scores)
                    select_tokens_count = int(len(q_token_scores)*self.query_term_ratio)
                    q_token_scores = q_token_scores[:select_tokens_count]
                    queries.append(Query(
                        id=q_id,
                        data=(" ".join([t for t, p in q_token_scores]), target_gz)
                    ))
                else:
                    queries.append(Query(
                        id=q_id,
                        data=(q_text, target_gz)
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
                            "fields": ["passage_text_tokenized"]
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