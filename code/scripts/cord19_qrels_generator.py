"""
Python script that parses the retrieval result to the qrels formate suitable for trec evaluation
"""
import argparse
from code.utils import io_util


def init_uid_metadata_map(uid_cord_id_info_path):
    uid_cord_id_info = io_util.read_json(uid_cord_id_info_path)
    uid_metadata_map = {}
    for doc_info in uid_cord_id_info:
        uid_metadata_map[doc_info['uid']] = {
            "doc_id": doc_info['paper_id'],
            'cord_uid': doc_info['cord_uid']
        }
    return uid_metadata_map


def write_to_qrels(uid_metadata_map, run_name, run_result_path, qrels_out_path):
    run_data = io_util.read_json(run_result_path)
    qrels_list = []

    for query in run_data:
        query_id = query['query_id']
        rank = 0
        for doc in query['relevant_docs']:
            if doc['doc_id'] not in uid_metadata_map:
                continue
            cord_id = uid_metadata_map[doc['doc_id']]['cord_uid']
            print(f"{query_id} Q0 {cord_id} {rank} {doc['score']:.4f} {run_name}")
            qrels_list.append(f"{query_id} Q0 {cord_id} {rank} {doc['score']:.4f} {run_name}")
            rank += 1
    io_util.write_list_to_file(qrels_out_path, qrels_list)


def main(args):
    run_result_path = args.run_result_path
    uid_cord_id_info = args.uid_cord_id_info
    run_name = args.run_name
    qrels_out_path = args.qrels_out_path

    uid_metadata_map = init_uid_metadata_map(uid_cord_id_info)

    write_to_qrels(uid_metadata_map, run_name, run_result_path, qrels_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Python script that parses the retrieval result to the qrels formate suitable for trec evaluation'
    )

    parser.add_argument('run_result_path', help='The path of the json output for the run.')
    parser.add_argument('uid_cord_id_info', help='The path of the uid and metadata info')
    parser.add_argument('run_name', help='The name of the run.')
    parser.add_argument('qrels_out_path', help='The name of the run.')

    args = parser.parse_args()
    main(args)
