import argparse

from lxml import etree

from code.indexing import es_index
from code.indexing.es_index import Index
from code.indexing.readers import CORD19Reader, CORD19ParagraphReader
from code.utils import io_util


def main(args):
    with open(args.query_file, 'r') as fp:
        topics = etree.parse(fp).getroot()
    queries = []
    for topic in topics:
        query = topic[0]
        assert query.tag == "query"
        queries.append({
            es_index.MAP_KEY__QUERY_ID: topic.attrib['number'],
            es_index.MAP_KEY__QUERY_TEXT: query.text
        })

    if args.index_type == "paragraphs":
        reader = CORD19ParagraphReader(batch_size=16384)
    else:
        reader = CORD19Reader(batch_size=1024)
    index = Index(args.index_name, host=args.host)

    print(f"Ranking {len(queries)} queries...")
    ranking_data = index.rank(queries=queries, size=100, query_builder=reader)
    io_util.write_json(args.output_file, ranking_data)

    print(ranking_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for testing the elasticsearch index in es_index'
    )

    parser.add_argument('host', help='Host address where the elasticsearch service is running')
    parser.add_argument('index_name', help='The name of the index')
    parser.add_argument('index_type', help='A choice between "paragraphs" or "doc"')
    parser.add_argument('query_file', help='File containing the CORD queries')
    parser.add_argument('output_file', help='Output path of the results in json.')

    args = parser.parse_args()
    main(args)
