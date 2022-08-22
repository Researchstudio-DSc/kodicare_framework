from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from elasticsearch.helpers import parallel_bulk
import json
import requests

# simple query template
def build_bool_query(query, size):
    data = {
        "from" : 0, 
        "size" : size,
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
        "from" : 0, 
        "size" : size,
        "query": {
            "multi_match" : {
                "query": query,
                "type": "cross_fields",
                "fields": fields
            }
        }
    }
    return data


class Index:
    
    def __init__(self, index_name, host="localhost:9200", es=None, ic=None, query_builder=build_multimatch_query):
        self.index_name = index_name
        self.host = host
        self.es = es if es != None else Elasticsearch(hosts=[host])
        self.ic = ic if ic != None else IndicesClient(self.es)
        self.query_builder = query_builder
    
    
    def create_index(self, index_body):
        """
        Create the index, recreate it if it exists
        """
        if self.ic.exists(index=self.index_name):
            self.ic.delete(index=self.index_name)
        self.ic.create(index=self.index_name, body=index_body)
    
    
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
        Index documents via iterator using the bulk helper
        """
        for doc_data in doc_iterator:
            for success, info in parallel_bulk(client=self.es, actions=self.action_gen(doc_data=doc_data)):
                if not success:
                    print('A document failed:', info)
    

    def rank(self, queries: list, retrieval_size: int=100):
        """
        Rank multiple queries simultaneously
        return rankings for each query
        """
        data_json = ""
        for query, fields in queries:
            q_header = {}
            query_data = self.query_builder(query, fields=fields, size=retrieval_size)
            data_json += json.dumps(q_header) + "\n"
            data_json += json.dumps(query_data) + "\n"
        headers = {
            'Content-type': 'application/json',
        }
        r = requests.post(f"http://{self.host}/{self.index_name}/_msearch", data=data_json, headers=headers)
        responses = r.json()['responses']
        ranking_data = []
        for ranking in responses:
            if "hits" not in ranking:
                print(ranking)
            ranking_data.append([hit['_source'] for hit in ranking["hits"]["hits"]])
        return ranking_data
