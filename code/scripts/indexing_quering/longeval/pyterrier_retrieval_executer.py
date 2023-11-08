"""
python script to generate runs for retrieval models and evaluate the models (the output is the evaluation)
"""

from code.indexing.pyterrier_indexer import *
import hydra


@hydra.main(version_base=None, config_path="../../../../conf", config_name=None)
def main(cfg):
    queries_path = cfg.test_collection.queries_path
    qrels_path = cfg.test_collection.qrels_path
    index_path = cfg.index.index_path

    evaluation_output_path = cfg.evaluation.evaluation_output_path

    queries_df = read_queries_longeval(queries_path)
    qrels_df = read_qrels_longeval(qrels_path)

    runs = {}
    for index, retrieval_model in enumerate(cfg.retrieval_models):
        model_info = retrieval_model[index]
        controls = {} if 'controls' not in model_info else model_info['controls']
        run_df = retrieve_run(index_path, queries_df, model_info['model'], controls=controls, num_results=100)
        runs[model_info['run_name']] = run_df

    evaluation_df = evaluate_run_set(runs, qrels_df, cfg.evaluation_metrics)

    # write the output
    evaluation_df.to_csv(evaluation_output_path, sep="\t")


if __name__ == '__main__':
    main()
