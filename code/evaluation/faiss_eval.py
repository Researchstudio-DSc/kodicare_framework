import argparse
from ast import arg
import json
from tkinter.messagebox import QUESTION
from lxml import etree
import os
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess

from code.indexing.faiss_index import Index
from code.indexing.readers import CORD19Reader, CORD19ParagraphReader

MODEL_FOLDER = "./models/doc2vec"

def doc_reranking(cord_ranking):
    # rerank the documents using the original ranking
    # but filter out duplicates
    found_ids = set()
    reranking = []
    for cord_uid, score in cord_ranking:
        if cord_uid in found_ids:
            continue
        found_ids.add(cord_uid)
        reranking.append((cord_uid, score))
    return reranking


def reciprocal_rank_score(rank):
    return 1/(10+rank)

def paragraph_reranking(cord_ranking):
    # rerank by calculating a score based on the rank positions of a
    # document's paragraphs in the original ranking
    cord_uid_scores = {}
    for rank, (cord_uid, _) in enumerate(cord_ranking):
        if cord_uid not in cord_uid_scores:
            cord_uid_scores[cord_uid] = 0
        cord_uid_scores[cord_uid] += reciprocal_rank_score(rank)
    return sorted(cord_uid_scores.items(), key=lambda x: x[1], reverse=True)


def main(args):

    with open(args.topics, 'r') as fp:
        topics = etree.parse(fp).getroot()

    model_path = os.path.join(MODEL_FOLDER, args.model_name)
    model = Doc2Vec.load(model_path)

    queries = []
    for topic in topics:
        query = topic[0]
        question = topic[1]
        assert query.tag == "query"
        assert question.tag == "question"
        processed_doc = simple_preprocess(f"{query.text} {question.text}")
        query_vector = model.infer_vector(processed_doc)
        queries.append(query_vector)
    
    with open(args.cord_id_title, "r") as fp:
        cord_id_title = json.load(fp)
    
    # mapping between IDs used in qrel_file and files in index
    cord_uid_mapping = {}
    for mapping in cord_id_title:
        # paper_id field can contain multiple ids separated by ';'
        paper_ids = [p.strip() for p in mapping['paper_id'].split(';')]
        for paper_id in paper_ids:
            cord_uid_mapping[paper_id] = mapping['cord_uid']
    

    if args.index_type == "paragraphs":
        reader = CORD19ParagraphReader(batch_size=16384)
    else:
        reader = CORD19Reader(batch_size=1024)
    index = Index(args.index_name)
    index.deserialize(args.index_folder)

    ranking_data = index.rank(queries=queries, size=args.size)

    for q_id, raw_ranking in enumerate(ranking_data):
        rank = 0
        cord_ranking = []
        # get general mapping to cord ids
        for entry, score in raw_ranking:
            #score = entry['_score']
            paper_id = entry['doc_id']
            cord_uid = cord_uid_mapping[paper_id]
            cord_ranking.append((cord_uid, score))
        
        if args.index_type == "paragraphs":
            reranking = paragraph_reranking(cord_ranking)
        else: # whole documents
            reranking = doc_reranking(cord_ranking)
        
        for rank, (cord_uid, score) in enumerate(reranking):
            print(f"{q_id+1} Q0 {cord_uid} {rank} {score:.4f} {args.run_name}")
    
    #with open(args.run_settings, "r") as fp:
    #    run_settings = json.load(fp)

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cord_id_title")
    parser.add_argument("--topics")
    parser.add_argument('--index_name', help='The name of the index')
    parser.add_argument('--index_folder', help='Folder to store the index')
    parser.add_argument('--index_type', help='paragraphs or doc')
    parser.add_argument('--run_name', help='Name of the run')
    parser.add_argument('--model_name', help='Name of the Doc2Vec model')
    parser.add_argument('--size', type=int, default=100, help='Number of results to retrieve')
    #parser.add_argument("--run_settings", help='File with the run settings')
    args = parser.parse_args()
    main(args)