import argparse
from ast import arg
import json
from lxml import etree

from code.indexing.es_index import Index
from code.indexing.readers import CORD19Reader, CORD19ParagraphReader


def main(args):

    with open(args.topics, 'r') as fp:
        topics = etree.parse(fp).getroot()
    queries = []
    for topic in topics:
        query = topic[0]
        assert query.tag == "query"
        queries.append(query.text)
    
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
    index = Index(args.index_name, host=args.host)

    ranking_data = index.rank(queries=queries, size=100, query_builder=reader)


    for q_id, raw_ranking in enumerate(ranking_data):
        found_ids = set()
        rank = 0
        for entry in raw_ranking:
            score = entry['_score']
            paper_id = entry['_source']['doc_id']
            cord_uid = cord_uid_mapping[paper_id]
            if cord_uid in found_ids:
                continue
            found_ids.add(cord_uid)
            print(f"{q_id+1} Q0 {cord_uid} {rank} {score} {args.run_name}")
            rank += 1
    
    #with open(args.run_settings, "r") as fp:
    #    run_settings = json.load(fp)

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Host address where the elasticsearch service is running')
    parser.add_argument("--cord_id_title")
    parser.add_argument("--topics")
    parser.add_argument('--index_name', help='The name of the index')
    parser.add_argument('--index_type', help='paragraphs or doc')
    parser.add_argument('--run_name', help='Name of the run')
    #parser.add_argument("--run_settings", help='File with the run settings')
    args = parser.parse_args()
    main(args)