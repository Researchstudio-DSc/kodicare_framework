from typing import Literal
import faiss
import pickle
import numpy as np
import os
from typing import List
from code.indexing.index_util import Query

class Index:

    # install faiss with:
    # $ conda install -c pytorch faiss-cpu
    # or
    # $ conda install -c pytorch faiss-gpu

    def __init__(self, index_name, index_folder, vector_size, mode: Literal["create", "load"]="load") -> None:
        # keep track of: 
        # - <faiss_index> containing the document vectors
        # - <index> containing orig. document ids and other data
        self.index_name = index_name
        self.index_folder = index_folder
        self.vector_size = vector_size
        self.faiss_index = None
        self.index = None
        self.mode = mode
        if mode == "create":
            self.create_index()
        if mode == "load":
            self.deserialize()
    

    def create_index(self):
        self.faiss_index = faiss.IndexFlatIP(self.vector_size)
        self.index = []
    

    def serialize(self, folder: str):
        faiss.write_index(self.faiss_index, os.path.join(folder, self.index_name+".faiss"))
        with open(os.path.join(folder, self.index_name+".pickle"), "wb") as fp:
            pickle.dump(self.index, fp)
    

    def deserialize(self):
        self.faiss_index = faiss.read_index(os.path.join(self.index_folder, self.index_name+".faiss"))
        with open(os.path.join(self.index_folder, self.index_name+".pickle"), "rb") as fp:
            self.index = pickle.load(fp)
    

    def index_docs(self, doc_iterator):
        for batch_data in doc_iterator:
            batch_doc_vectors = []
            for uid, doc_id, doc_vector in batch_data:
                batch_doc_vectors.append(doc_vector)
                document_obj = {
                    "uid": uid, # doc_uid or paragraph_id
                    "doc_id": doc_id # CORD19 doc_id
                }
                self.index.append(document_obj)
            batch_doc_vectors = np.array(batch_doc_vectors).astype("float32")
            self.faiss_index.add(batch_doc_vectors)
    
    

    def rank(self, queries: List[Query], size=100) -> List[Query]:
        query_vectors = np.array([q.data for q in queries])
        # retrieve size results for the query
        scores, faiss_doc_ids = self.faiss_index.search(query_vectors, size)
        ranking_data = []
        for faiss_doc_id, score, query in zip(faiss_doc_ids, scores, queries):
            query_results = []
            for i, s in zip(faiss_doc_id, score):
                query_results.append((self.index[int(i)], s))
            ranking_data.append(Query(
                id=query.id,
                data=query.data,
                relevant_docs=query_results
            ))
        return ranking_data