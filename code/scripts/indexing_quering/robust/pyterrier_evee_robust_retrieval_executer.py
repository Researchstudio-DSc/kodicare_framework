"""
python script to generate runs for retrieval models and evaluate the models (the output is the evaluation)
"""

import hydra

from code.indexing.pyterrier_indexer import *

if not pt.started():
    pt.init(logging='TRACE', boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])


@hydra.main(version_base=None, config_path="../../../../conf", config_name=None)
def main(cfg):
    index_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.index.index_path))
    print(index_path)

    evaluation_perquery_output_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir,
                                                                     cfg.dtc.evaluation.evaluation_perquery_output_path))
    evaluation_mean_output_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir,
                                                                 cfg.dtc.evaluation.evaluation_mean_output_path))

    if not path_exits(evaluation_perquery_output_path):
        mkdir(evaluation_perquery_output_path)
    if not path_exits(evaluation_mean_output_path):
        mkdir(evaluation_mean_output_path)

    queries_df = pt.get_dataset("trec-robust-2004").get_topics()
    qrels_df = pt.get_dataset("trec-robust-2004").get_qrels()

    ees_indices = list_directories(index_path)

    for ee_index in ees_indices:
        runs = {}
        for index, retrieval_model in enumerate(cfg.retrieval_models):
            model_info = retrieval_model[index]
            controls = {} if 'controls' not in model_info else model_info['controls']
            run_df = retrieve_run(join(index_path, ee_index), queries_df, model_info['model'], controls=controls, num_results=100)
            runs[model_info['run_name']] = run_df

        evaluation_mean_df = evaluate_run_set(runs, qrels_df, cfg.evaluation_metrics, perquery=False)
        evaluation_perquery_df = evaluate_run_set(runs, qrels_df, cfg.evaluation_metrics, perquery=True)

        # write the output
        evaluation_mean_df.to_csv(join(evaluation_mean_output_path, ee_index + '.csv'), sep="\t")
        evaluation_perquery_df.to_csv(join(evaluation_perquery_output_path, ee_index + '.csv'), sep="\t")


if __name__ == '__main__':
    main()
