"""
python script to generate runs for retrieval models and evaluate the models (the output is the evaluation)
"""

import hydra

from code.indexing.pyterrier_indexer import *


@hydra.main(version_base=None, config_path="../../../../conf", config_name=None)
def main(cfg):
    queries_path = cfg.test_collection.queries_path
    qrels_path = cfg.test_collection.qrels_path
    index_path = cfg.index.index_path

    evaluation_perquery_output_path = cfg.evaluation.evaluation_perquery_output_path
    evaluation_mean_output_path = cfg.evaluation.evaluation_mean_output_path

    queries_df = read_queries_longeval_trec(queries_path)
    qrels_df = read_qrels_longeval(qrels_path)

    runs = {}
    for index, retrieval_model in enumerate(cfg.retrieval_models):
        model_info = retrieval_model[index]
        controls = {} if 'controls' not in model_info else model_info['controls']
        run_df = retrieve_run(index_path, queries_df, model_info['model'], controls=controls, num_results=100)
        runs[model_info['run_name']] = run_df

    evaluation_mean_df = evaluate_run_set(runs, qrels_df, cfg.evaluation_metrics, perquery=False)
    evaluation_perquery_df = evaluate_run_set(runs, qrels_df, cfg.evaluation_metrics, perquery=True)

    # write the output
    evaluation_mean_df.to_csv(evaluation_mean_output_path, sep="\t")
    evaluation_perquery_df.to_csv(evaluation_perquery_output_path, sep="\t")


if __name__ == '__main__':
    main()
