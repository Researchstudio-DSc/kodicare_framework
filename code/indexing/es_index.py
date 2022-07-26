from typing import Literal
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
import json
import requests
from typing import List
from code.indexing.index_util import Query

MAP_KEY__QUERY_TEXT = "query_text"
MAP_KEY__QUERY_ID = "query_id"
MAP_KEY__RELEVANT_DOCS = "relevant_docs"
MAP_KEY__DOC_ID = "doc_id"
MAP_KEY__UID = "uid"
MAP_KEY__SCORE = "score"
MAP_KEY__INFO = "info"


# simple query template
def build_bool_query(query, size):
    data = {
        "from": 0,
        "size": size,
        "query": {
            "bool": {
                "should": {
                    "match": {
                        "text": query
                    }
                }
            }
        }
    }
    return data


# simple query template
def build_multimatch_query(query, fields, size):
    data = {
        "from": 0,
        "size": size,
        "query": {
            "multi_match": {
                "query": query,
                "type": "cross_fields",
                "fields": fields
            }
        }
    }
    return data


class Index:

    def __init__(self, index_name, host="localhost:9200", index_body_path=None, 
    mode: Literal["create", "load"]="load", es=None, ic=None):
        self.index_name = index_name
        self.host = host
        self.index_body_path = index_body_path
        self.mode = mode
        self.es = es if es != None else Elasticsearch(hosts=[host])
        self.ic = ic if ic != None else IndicesClient(self.es)
        if mode == "create":
            with open(index_body_path, 'r') as fp:
                self.index_body = json.load(fp)
            self.create_index()

    def create_index(self):
        """
        Create the index, recreate it if it exists
        """
        if self.ic.exists(index=self.index_name):
            self.ic.delete(index=self.index_name)
        self.ic.create(index=self.index_name, body=self.index_body)

    def delete_index(self):
        """
        Delete the index
        """
        if self.ic.exists(index=self.index_name):
            self.ic.delete(index=self.index_name)

    def update_settings(self, index_body):
        """
        Update settings of the index.
        First close the index, and open it again after the update
        """
        settings = index_body['settings']
        if "number_of_shards" in settings["index"]:
            del settings["index"]["number_of_shards"]
        if "number_of_replicas" in settings["index"]:
            del settings["index"]["number_of_replicas"]
        self.ic.close(index=self.index_name)
        self.ic.put_settings(settings=settings, index=self.index_name)
        self.ic.open(index=self.index_name)

    def index_exists(self):
        """
        Return True if index exists
        """
        return self.ic.exists(self.index_name)

    def action_gen(self, doc_data):
        for passage_body in doc_data:
            yield {
                '_index': self.index_name,
                '_source': passage_body
            }

    def index_docs(self, doc_iterator):
        """
        Index documents via iterator using the bulk function
        """
        for batch_data in doc_iterator:
            data_json = ""
            for document_id, doc_source in batch_data:
                action_and_meta_data = {"index": {"_index": self.index_name, "_id": document_id}}
                data_json += json.dumps(action_and_meta_data) + "\n"
                data_json += json.dumps(doc_source) + "\n"
            headers = {
                'Content-type': 'application/json',
            }
            r = requests.post(f"http://{self.host}/_bulk", data=data_json, headers=headers)
            resp = r.json()

    def rank(self, queries: List[Query], size=100, query_builder=None) -> List[Query]:
        """
        Rank multiple queries simultaneously
        return rankings for each query
        """
        data_json = ""
        for query in queries:
            q_header = {}
            query_data = query_builder.build(query.data, size=size)
            data_json += json.dumps(q_header) + "\n"
            data_json += json.dumps(query_data) + "\n"
        headers = {
            'Content-type': 'application/json',
        }
        r = requests.post(f"http://{self.host}/{self.index_name}/_msearch", data=data_json, headers=headers)
        responses = r.json()['responses']
        ranking_data = []
        for q_i, ranking in enumerate(responses):
            if "hits" not in ranking:
                print(ranking)
            query_info = queries[q_i]
            """ranking_data.append({
                MAP_KEY__QUERY_TEXT: query_info.text,
                MAP_KEY__QUERY_ID: query_info.id,
                MAP_KEY__RELEVANT_DOCS:
                    [{MAP_KEY__SCORE: hit['_score'], MAP_KEY__INFO: hit['_source']} for hit in ranking["hits"]["hits"]]
            })"""
            ranking_data.append(Query(
                id=query_info.id,
                data=query_info.data,
                relevant_docs=[{Query.KEY_SCORE: hit['_score'], Query.KEY_SOURCE: hit['_source']} for hit in ranking["hits"]["hits"]]
            ))
        return ranking_data
