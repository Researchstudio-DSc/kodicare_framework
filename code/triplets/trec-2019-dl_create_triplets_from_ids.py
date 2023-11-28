import os
import json
import argparse
from tqdm import tqdm
import time
import random
random.seed(42)


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


def main(args):
    n_docs = args.n_docs
    document_file = args.document_file
    queries_file = args.queries_file
    triplet_id_file = args.triplet_id_file
    triplet_text_file = args.triplet_text_file
    truncate = args.truncate
    analysis = args.analysis
    upper_bound=args.upper_bound
    lower_bound=args.lower_bound

    documents = get_documents(document_file, n_docs=n_docs)
    queries = get_queries(queries_file)

    triplets = []

    print("Creating Triplets")
    with open(triplet_id_file, 'r') as fp:
        for line in fp:
            q_id, pos_doc_id, neg_doc_id, s = line.strip().split()
            s = float(s)
            if s > upper_bound or s < lower_bound:
                continue
            triplets.append((q_id, pos_doc_id, neg_doc_id))

    # shuffle training data
    random.shuffle(triplets)

    with open(triplet_text_file, 'w') as fp_out:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create Triplets for dense retrieval training in the format: query-text<tab>pos-text<tab>neg-text.'
    )

    parser.add_argument('--document_file', help='TSV file with document data')
    parser.add_argument('--queries_file', help='File with queries data')
    parser.add_argument('--triplet_id_file', help='File with triplet ids')
    parser.add_argument('--triplet_text_file', help='Out File with triplet texts')
    parser.add_argument('--upper_bound', default=1.00, type=float, help='Upper score limit')
    parser.add_argument('--lower_bound', default=0.00, type=float, help='Lower score limit')
    parser.add_argument('-t', '--truncate',
                    action='store_true')
    parser.add_argument('-a', '--analysis',
                    action='store_true')
    parser.add_argument('--n_docs', default=3213835, type=int, help='Number of documents')

    args = parser.parse_args()
    main(args)