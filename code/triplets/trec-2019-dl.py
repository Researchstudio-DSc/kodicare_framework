import os
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import re
import numpy as np
from tqdm import tqdm
import pysparnn.cluster_index as ci
import time
import hydra
from omegaconf import DictConfig

from kd_models import TFIDFDelta, BERTDelta
from indexing import FaissHNSWIndexer, PySparnnIndexer
no_num_clean_p = re.compile(r'[^\w\s]+|\d+', re.UNICODE)

# Create Triplets for dense retrieval training.

VECTOR_SPARSE = "vector_sparse"
VECTOR_DENSE = "vector_dense"


def document_iterator(document_file, n_docs, batch_size:int=None):
    """
    Iterate through documents, returning only their text.
    Yields single documents if batch_size is zero or none.
    Yields lists of documents if batch_size is set.
    """
    with open(document_file, "r") as fp:
        if not batch_size or batch_size <= 0: # single document version
            for doc_line in tqdm(fp, total=n_docs):
                data = doc_line.strip().split(sep="\t")
                if len(data) != 4:
                    #print(len(data), data)
                    continue
                doc_id, doc_url, doc_title, doc_text = data
                yield doc_text
        else: # batched version
            docs_batch = []
            for doc_line in tqdm(fp, total=n_docs):
                data = doc_line.strip().split(sep="\t")
                if len(data) != 4:
                    #print(len(data), data)
                    continue
                doc_id, doc_url, doc_title, doc_text = data
                docs_batch.append(doc_text)
                if len(docs_batch) >= batch_size:
                    yield docs_batch
                    docs_batch = []
            if len(docs_batch) > 0:
                yield docs_batch


def get_document_ids(document_file):
    doc_ids = []
    doc_ids_inv = {}
    doc_ids_int64 = []
    with open(document_file, "r") as fp:
        for doc_line in fp:
            data = doc_line.strip().split(sep="\t")
            if len(data) != 4:
                #print(len(data), data)
                continue
            doc_id, doc_url, doc_title, doc_text = data
            doc_ids_inv[doc_id] = len(doc_ids)
            doc_ids_int64.append(len(doc_ids))
            doc_ids.append(doc_id)
    return doc_ids, doc_ids_inv, doc_ids_int64


def read_qrels(qrels_file):
    relevance_judgements = {}
    with open(qrels_file, "r") as fp:
        for qrel_line in fp:
            q_id, _, doc_id, relevance = qrel_line.strip().split()
            relevance = int(relevance)
            if relevance == 1:
                if q_id not in relevance_judgements:
                    relevance_judgements[q_id] = set()
                relevance_judgements[q_id].add(doc_id)
    return relevance_judgements


def batched_search(cp, search_features_vec, batch_size=4096, k=10, k_clusters=2, return_distance=True):
    search_features_vec_size = search_features_vec.shape[0]
    steps = int(np.ceil(search_features_vec_size / batch_size))
    sims_with_idx = []
    for i in tqdm(range(steps)):
        start_idx = i*batch_size
        end_idx = start_idx + batch_size
        sims_with_idx.extend(
            cp.search(search_features_vec[start_idx:end_idx,:], k=k, 
                      k_clusters=k_clusters, return_distance=return_distance)
        )
    return sims_with_idx


def relevance_judgement_batches_iter(embeddings, relevance_judgements, doc_ids_inv, batch_size, vector_type):
    """
    Iterate through relevance judgements in batches and return positive document embeddings in batches
    """
    search_features_vec_batch = []
    doc_data_batch = []

    for q_id, relevant_doc_ids in tqdm(relevance_judgements.items()):
        for pos_doc_id in relevant_doc_ids:
            if pos_doc_id not in doc_ids_inv:
                continue
            relevant_doc_vector = embeddings[doc_ids_inv[pos_doc_id]]
            search_features_vec_batch.append(relevant_doc_vector)
            doc_data_batch.append((q_id, pos_doc_id, relevant_doc_ids))
            if len(search_features_vec_batch) >= batch_size:
                if vector_type == VECTOR_SPARSE:
                    yield scipy.sparse.vstack(search_features_vec_batch), doc_data_batch
                elif vector_type == VECTOR_DENSE:
                    yield np.vstack(search_features_vec_batch), doc_data_batch
                else:
                    assert False
                search_features_vec_batch = []
                doc_data_batch = []
    if len(search_features_vec_batch) > 0:
        if vector_type == VECTOR_SPARSE:
            yield scipy.sparse.vstack(search_features_vec_batch), doc_data_batch
        elif vector_type == VECTOR_DENSE:
            yield np.vstack(search_features_vec_batch), doc_data_batch
        else:
            assert False


@hydra.main(version_base=None, config_path=".", config_name=None)
def main(cfg):
    n_docs = cfg.n_docs
    document_file = cfg.document_file
    qrels_file = cfg.qrels_file
    triplet_folder = cfg.triplet_folder
    kd_experiment_name = cfg.kd_experiment_name
    kd_model_config = cfg.kd_model
    iterator_batch_size = cfg.kd_model.iterator_batch_size
    kd_model_type = cfg.kd_model.model_type
    index_config = cfg.index_config
    search_index = cfg.index_config.search_index
    top_k = cfg.index_config.top_k
    batch_size = cfg.index_config.batch_size

    triplet_id_file = os.path.join(triplet_folder, f"triplets_{kd_experiment_name}.train.id")

    print("Creating Vectors")

    if kd_model_type == "tfidf":
        model = TFIDFDelta(kd_model_config)
        vector_type = VECTOR_SPARSE
    elif kd_model_type == "bert-embedding":
        model = BERTDelta(kd_model_config)
        vector_type = VECTOR_DENSE
    else:
        print(f"KD Model {kd_model_type} is not known.")


    t0 = time.time()
    embeddings = model.create_embeddings(document_iterator(document_file, n_docs, iterator_batch_size))
    t1 = time.time()
    print("Took", t1-t0)

    doc_ids, doc_ids_inv, doc_ids_int64 = get_document_ids(document_file)
    relevance_judgements = read_qrels(qrels_file)

    print("Creating Search Index")
    t0 = time.time()
    if index_config["search_index"] == "pysparnn":
        assert vector_type == VECTOR_SPARSE
        indexer = PySparnnIndexer(index_config)
    elif index_config["search_index"] == "faiss":
        assert vector_type == VECTOR_DENSE
        indexer = FaissHNSWIndexer(index_config)
    else:
        print(f"Search index {search_index} is not known.")
    indexer.index(doc_ids_int64, embeddings)
    t1 = time.time()
    print("Took", t1-t0)

    print("Creating Triplets")
    with open(triplet_id_file, 'w') as fp:
        buffer = []
        for search_features_vec_batch, doc_data_batch in relevance_judgement_batches_iter(embeddings, relevance_judgements, doc_ids_inv, batch_size, vector_type):
            # search approximate nearest neighbors
            #sims_with_idx = indexer.search(search_features_vec_batch, k=top_k, k_clusters=k_clusters, return_distance=True)
            sims_with_idx = indexer.search(search_features_vec_batch, top_k=top_k)

            # create triplets
            for i in range(len(doc_data_batch)):
                # set all known relevant documents to -inf
                q_id, pos_doc_id, relevant_doc_ids = doc_data_batch[i]
                for similarity, neg_doc_id in sims_with_idx[i]:
                    neg_doc_id =doc_ids[int(neg_doc_id)]
                    if neg_doc_id in relevant_doc_ids:
                        continue
                    buffer.append(f"{q_id} {pos_doc_id} {neg_doc_id} {similarity:.4f}\n")
                    if len(buffer) >= 10000:
                        fp.write("".join(buffer))
                        buffer = []
        if len(buffer) > 0:
            fp.write("".join(buffer))
    


if __name__ == '__main__':
    main()