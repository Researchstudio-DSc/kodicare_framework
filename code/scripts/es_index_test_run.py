import argparse
import json

from code.indexing.es_index import Index


def main(args):
    # index body
    with open(args.index_body, 'r') as fp:
        index_body = json.load(fp)
    
    index_settings = index_body['settings']
    
    # test data
    with open(args.data_json, 'r') as fp:
        data = [json.load(fp)]

    index = Index(index_settings["index_name"], host=args.host)

    index.create_index(index_body=index_body)
    index.index_docs(data)
    ranking_data = index.rank(queries=["vaccine"])
    print(ranking_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for testing the elasticsearch index in es_index'
    )

    parser.add_argument('host', help='Host address where the elasticsearch service is running')
    parser.add_argument('index_body', help='The the index_body file containing settings and mappings')
    parser.add_argument('data_json', help='File containing the data that should be indexed')

    args = parser.parse_args()
    main(args)