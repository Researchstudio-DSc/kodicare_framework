"""
python script to generate runs for retrieval models and evaluate the models (the output is the evaluation)
"""

import hydra

from code.indexing.pyterrier_indexer import *
import pandas as pd


@hydra.main(version_base=None, config_path="../../../../conf", config_name=None)
def main(cfg):
    runs_dir = join(cfg.config.root_dir, cfg.test_collection.runs_output_dir)
    print(runs_dir)
    dtc_ids = read_pickle(join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evee_info_path)))
    print(join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evee_info_path))
    evaluation_perquery_output_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir,
                                                                     cfg.dtc.evaluation.evaluation_perquery_output_path))
    evaluation_mean_output_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir,
                                                                 cfg.dtc.evaluation.evaluation_mean_output_path))

    if not path_exits(evaluation_perquery_output_path):
        mkdir(evaluation_perquery_output_path)
    if not path_exits(evaluation_mean_output_path):
        mkdir(evaluation_mean_output_path)

    qrels_df = pt.get_dataset("trec-robust-2004").get_qrels()

    for (ee_index, tc_ids) in enumerate(dtc_ids):
        runs = {}
        for index, retrieval_model in enumerate(cfg.retrieval_models):
            model_info = retrieval_model[index]
            merged_run_df = pd.read_csv(join(runs_dir, 'run_' + model_info['run_name'] + '.csv'))
            mask = merged_run_df['docno'].isin(tc_ids)
            run_df = merged_run_df[mask]
            runs[model_info['run_name']] = run_df

        evaluation_mean_df = evaluate_run_set(runs, qrels_df, cfg.evaluation_metrics, perquery=False)
        evaluation_perquery_df = evaluate_run_set(runs, qrels_df, cfg.evaluation_metrics, perquery=True)

        # write the output
        evaluation_mean_df.to_csv(join(evaluation_mean_output_path, 'tc_' + str(ee_index) + '.csv'), sep="\t")
        evaluation_perquery_df.to_csv(join(evaluation_perquery_output_path, 'tc_' + str(ee_index) + '.csv'), sep="\t")


if __name__ == '__main__':
    main()
