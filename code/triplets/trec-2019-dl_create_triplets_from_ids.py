import os
import json
import argparse
from tqdm import tqdm
import time
import random
import hydra
from omegaconf import DictConfig


# Create Triplets for dense retrieval training in the format: query-text<tab>pos-text<tab>neg-text.


def get_queries(queries_file):
    queries = {}
    with open(queries_file, "r") as fp:
        for queries_line in fp:
            q_id, q_text = queries_line.strip().split(sep="\t")
            queries[q_id] = q_text
    return queries


def get_documents(document_file, n_docs):
    documents = {}
    with open(document_file, "r") as fp:
        for doc_line in tqdm(fp, total=n_docs):
            data = doc_line.strip().split(sep="\t")
            if len(data) != 4:
                #print(len(data), data)
                continue
            doc_id, doc_url, doc_title, doc_text = data
            documents[doc_id] = doc_text
    return documents


def create_triplets(cfg, triplet_slice_name, documents, queries, upper_bound, lower_bound):
    random_seed = cfg.random_seed
    triplet_text_folder = cfg.triplet_text_folder
    triplet_id_file = cfg.triplet_id_file
    truncate = cfg.truncate
    analysis = cfg.analysis

    random.seed(random_seed)

    

    triplets = []

    print(f"Creating Triplets {triplet_slice_name}")
    with open(triplet_id_file, 'r') as fp:
        for line in fp:
            q_id, pos_doc_id, neg_doc_id, s = line.strip().split()
            s = float(s)
            if s > upper_bound or s < lower_bound:
                continue
            triplets.append((q_id, pos_doc_id, neg_doc_id))

    # shuffle training data
    random.shuffle(triplets)

    with open(os.path.join(triplet_text_folder, triplet_slice_name), 'w') as fp_out:
        for q_id, pos_doc_id, neg_doc_id in tqdm(triplets):
            pos_doc = documents[pos_doc_id]
            neg_doc = documents[neg_doc_id]
            if truncate:
                pos_doc = " ".join(pos_doc.split()[:512])
                neg_doc = " ".join(neg_doc.split()[:512])
            if analysis:
                fp_out.write(f"{queries[q_id]}\n===\n{pos_doc}\n===\n{neg_doc}\t{s}\n\n\n")
            else:
                fp_out.write(f"{queries[q_id]}\t{pos_doc}\t{neg_doc}\n")

@hydra.main(version_base=None, config_path=".", config_name=None)
def main(cfg: DictConfig):
    triplet_files = cfg.triplet_files
    n_docs = cfg.n_docs
    document_file = cfg.document_file
    queries_file = cfg.queries_file

    documents = get_documents(document_file, n_docs=n_docs)
    queries = get_queries(queries_file)

    for triplet_slice_name, kwargs in triplet_files.items():
        create_triplets(cfg=cfg, triplet_slice_name=triplet_slice_name, documents=documents, queries=queries, **kwargs)

if __name__ == '__main__':
    main()