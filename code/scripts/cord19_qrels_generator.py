"""
Python script that parses the retrieval result to the qrels formate suitable for trec evaluation
"""

import hydra

from code.utils import io_util


def write_to_qrels(run_name, run_result_path, qrels_out_path):
    run_data = io_util.read_json(run_result_path)
    qrels_list = []

    for query in run_data:
        query_id = query['query_id']
        rank = 0
        for doc in query['relevant_docs']:
            cord_id = doc['cord_uid']
            print(f"{query_id} Q0 {cord_id} {rank} {doc['score']:.4f} {run_name}")
            qrels_list.append(f"{query_id} Q0 {cord_id} {rank} {doc['score']:.4f} {run_name}")
            rank += 1
    io_util.write_list_to_file(qrels_out_path, qrels_list)


@hydra.main(version_base=None, config_path="../../conf", config_name="cord19_config")
def main(cfg):
    run_result_path = io_util.join(cfg.config.root_dir, io_util.join(cfg.config.output_dir,
                                                                     cfg.retrieval.rerank_path))
    run_name = cfg.retrieval.run_name
    qrels_out_path = io_util.join(cfg.config.root_dir, io_util.join(cfg.config.output_dir,
                                                                    cfg.retrieval.run_name + '.txt'))

    write_to_qrels(run_name, run_result_path, qrels_out_path)


if __name__ == '__main__':
    main()

