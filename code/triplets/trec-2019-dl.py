import os
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from tqdm import tqdm
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


def get_top_k_similar(relevant_doc_vector, doc_vecs, relevant_doc_ids, doc_ids, doc_ids_inv, top_k=10, upper_bound=1.0, lower_bound=0.70):
    # calculate similarity
    sim = cosine_similarity(relevant_doc_vector, doc_vecs)
    # set all known relevant documents to -inf
    relevant_doc_idx = [doc_ids_inv[doc_id] for doc_id in relevant_doc_ids if doc_id in doc_ids_inv]
    sim[0,relevant_doc_idx] = -np.inf
    # set all documents outside the boundaries to -inf
    sim = np.where(np.logical_and(sim <= upper_bound, sim >= lower_bound), sim, -np.inf)
    # get the top k document ids with the highest similarity, filtering out -inf
    top_sims_idx = np.argpartition(sim[0,:], -top_k)[-top_k:]
    sims_with_idx = [(doc_ids[idx], s) for idx, s in zip(top_sims_idx, sim[0,:][top_sims_idx]) 
                     if s != -np.inf]
    return sims_with_idx



def main(args):
    n_docs = args.n_docs
    document_file = args.document_file
    qrels_file = args.qrels_file
    triplet_id_file = args.triplet_id_file
    print("Creating Vectors")
    tfidf_vect = TfidfVectorizer(max_df=0.75, min_df=10)
    x = tfidf_vect.fit_transform(document_iterator(document_file, n_docs, only_text=True))

    doc_ids, doc_ids_inv = get_document_ids(document_file)
    relevance_judgements = read_qrels(qrels_file)

    print("Creating Triplets")
    with open(triplet_id_file, 'w') as fp:
        for q_id, relevant_docs in tqdm(relevance_judgements.items()):
            for pos_doc_id in relevant_docs:
                if pos_doc_id not in doc_ids_inv:
                    continue
                relevant_doc_vector = x[doc_ids_inv[pos_doc_id]]
                sims_with_idx = get_top_k_similar(relevant_doc_vector=relevant_doc_vector, doc_vecs=x, 
                                                doc_ids=doc_ids, doc_ids_inv=doc_ids_inv, relevant_doc_ids=relevant_docs, lower_bound=0.40)
                sims_with_idx = sorted(sims_with_idx, key=lambda x:x[1], reverse=True)
                for neg_doc_id, s in sims_with_idx:
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