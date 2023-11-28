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
no_num_clean_p = re.compile(r'[^\w\s]+|\d+', re.UNICODE)

n_docs = 3213835
document_file = "/home/tfink/data/kodicare/trec-2019-dl/doc_ret/msmarco-docs.head10k.tsv"
queries_file = "/home/tfink/data/kodicare/trec-2019-dl/doc_ret/msmarco-doctrain-queries.tsv"
qrels_file = "/home/tfink/data/kodicare/trec-2019-dl/doc_ret/msmarco-doctrain-qrels.tsv"
triplet_id_file = "/home/tfink/data/kodicare/trec-2019-dl/doc_ret/triplets.txt"


def document_iterator(document_file, n_docs, only_text=False):
    with open(document_file, "r") as fp:
        for doc_line in tqdm(fp, total=n_docs):
            data = doc_line.strip().split(sep="\t")
            if len(data) != 4:
                #print(len(data), data)
                continue
            doc_id, doc_url, doc_title, doc_text = data
            if only_text:
                yield doc_text
            else:
                yield doc_id, doc_url, doc_title, doc_text


def get_document_ids(document_file):
    doc_ids = []
    doc_ids_inv = {}
    with open(document_file, "r") as fp:
        for doc_line in fp:
            data = doc_line.strip().split(sep="\t")
            if len(data) != 4:
                #print(len(data), data)
                continue
            doc_id, doc_url, doc_title, doc_text = data
            doc_ids_inv[doc_id] = len(doc_ids)
            doc_ids.append(doc_id)
    return doc_ids, doc_ids_inv


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


def read_queries():
    queries = {}
    with open(queries_file, "r") as fp:
        for queries_line in fp:
            q_id, q_text = queries_line.strip().split(sep="\t")
            queries[q_id] = q_text
    return queries

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


def relevance_judgement_batches_iter(x, relevance_judgements, doc_ids_inv, batch_size):
    search_features_vec_batch = []
    doc_data_batch = []

    for q_id, relevant_doc_ids in tqdm(relevance_judgements.items()):
        for pos_doc_id in relevant_doc_ids:
            if pos_doc_id not in doc_ids_inv:
                continue
            relevant_doc_vector = x[doc_ids_inv[pos_doc_id]]
            search_features_vec_batch.append(relevant_doc_vector)
            doc_data_batch.append((q_id, pos_doc_id, relevant_doc_ids))
            if len(search_features_vec_batch) >= batch_size:
                yield scipy.sparse.vstack(search_features_vec_batch), doc_data_batch
                search_features_vec_batch = []
                doc_data_batch = []
    if len(search_features_vec_batch) > 0:
        yield scipy.sparse.vstack(search_features_vec_batch), doc_data_batch


def main(args):
    n_docs = args.n_docs
    document_file = args.document_file
    qrels_file = args.qrels_file
    triplet_id_file = args.triplet_id_file
    top_k = 100
    k_clusters = 5
    batch_size = 1000
    upper_bound=1.0
    lower_bound=0.40
    print("Creating Vectors")
    t0 = time.time()
    tfidf_vect = TfidfVectorizer(max_df=0.75, min_df=10)
    x = tfidf_vect.fit_transform(document_iterator(document_file, n_docs, only_text=True))
    t1 = time.time()
    print("Took", t1-t0)

    doc_ids, doc_ids_inv = get_document_ids(document_file)
    relevance_judgements = read_qrels(qrels_file)

    print("Creating Search Index")
    t0 = time.time()
    cp = ci.MultiClusterIndex(x, doc_ids)
    t1 = time.time()
    print("Took", t1-t0)

    print("Creating Triplets")
    with open(triplet_id_file, 'w') as fp:
        for search_features_vec_batch, doc_data_batch in relevance_judgement_batches_iter(x, relevance_judgements, doc_ids_inv, batch_size):
            # search approximate nearest neighbors
            sims_with_idx = cp.search(search_features_vec_batch, k=top_k, k_clusters=k_clusters, return_distance=True)

            # create triplets
            for i in range(len(doc_data_batch)):
                # set all known relevant documents to -inf
                q_id, pos_doc_id, relevant_doc_ids = doc_data_batch[i]
                for distance, neg_doc_id in sims_with_idx[i]:
                    s = 1-float(distance)
                    if neg_doc_id in relevant_doc_ids or s > upper_bound or s < lower_bound:
                        continue
                    fp.write(f"{q_id} {pos_doc_id} {neg_doc_id} {s:.4f}\n")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create Triplets for dense retrieval training.'
    )

    parser.add_argument('--document_file', help='TSV file with document data')
    parser.add_argument('--qrels_file', help='File with TREC qrel data')
    parser.add_argument('--triplet_id_file', help='Output file for the triplets')
    parser.add_argument('--n_docs', default=3213835, type=int, help='Number of documents')

    args = parser.parse_args()
    main(args)