import argparse
import json
import time

import code
import code.indexing
from code.indexing.es_index import Index
from code.indexing.readers import CORD19Reader, CORD19ParagraphReader


def main(args):
    # index body
    with open(args.index_body, 'r') as fp:
        index_body = json.load(fp)

    if args.index_type == "paragraphs":
        reader = CORD19ParagraphReader(batch_size=16384)
    else:
        reader = CORD19Reader(batch_size=1024)
    index = Index(args.index_name, host=args.host)

    index.create_index(index_body=index_body)
    index.index_docs(reader.iterate(args.data_folder))
    time.sleep(1)
    ranking_data = index.rank(queries=[("vaccine", ["paragraph_text"])])
    #print(ranking_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for testing the elasticsearch index in es_index'
    )

    parser.add_argument('host', help='Host address where the elasticsearch service is running')
    parser.add_argument('index_name', help='The name of the index')
    parser.add_argument('index_type', help='paragraphs or doc')
    parser.add_argument('index_body', help='The the index_body file containing settings and mappings')
    parser.add_argument('data_folder', help='Folder containing the files that should be indexed')

    args = parser.parse_args()
    main(args)