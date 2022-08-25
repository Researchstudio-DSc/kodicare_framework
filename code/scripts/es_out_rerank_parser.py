"""
a python script to parse the output from the elastic search ranking to the formate of the reranking input
"""

import argparse
from code.retrieval.reranking import delta_reranking_interface
from code.indexing import es_index
from code.utils import io_util


def main(args):
    es_data = io_util.read_json(args.es_output_path)
    rerank_formate = []

    for query_result in es_data:
        relevant_docs = []
        for doc in query_result[es_index.MAP_KEY__RELEVANT_DOCS]:
            # print(doc)
            relevant_docs.append({
                delta_reranking_interface.MAP_KEY__DOC_ID: doc[es_index.MAP_KEY__INFO][es_index.MAP_KEY__DOC_ID],
                delta_reranking_interface.MAP_KEY__SCORE: doc[es_index.MAP_KEY__SCORE],
            })
        rerank_formate.append({
            delta_reranking_interface.MAP_KEY__QUERY: query_result[es_index.MAP_KEY__QUERY_TEXT],
            delta_reranking_interface.MAP_KEY__QUERY_ID: query_result[es_index.MAP_KEY__QUERY_ID],
            delta_reranking_interface.MAP_KEY__RELEVANT_DOCS: relevant_docs
        })

    io_util.write_json(args.rerank_formate_output, rerank_formate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to change the formate of the es_index output to the reranking input'
    )

    parser.add_argument('es_output_path', help='The path of the output from the es_index.')
    parser.add_argument('rerank_formate_output', help='The path of the json file for the rerank formate.')

    args = parser.parse_args()
    main(args)
