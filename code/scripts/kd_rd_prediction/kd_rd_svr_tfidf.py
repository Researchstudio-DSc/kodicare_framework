import hydra
import pandas as pd

from code.kd_rd_predition import kd_rd_svr
from code.knowledge_delta import tfidf_docs_kd
from code.utils.io_util import *


def construct_rd_labels(rd_dir, start_step, end_step, epochs, systems, metrics):
    res_change_map = init_rd_map(systems, metrics)

    for step in range(start_step, end_step):
        for tc_ind1 in range(0, epochs - step):
            for tc_ind2 in range(tc_ind1 + step, epochs - step):
                print('RD for', tc_ind1, tc_ind2, step)
                if tc_ind2 >= epochs:
                    break
                update_res_change_map(rd_dir, res_change_map, tc_ind1, tc_ind2)

    rd_df = pd.DataFrame.from_dict(res_change_map)
    rd_df['tc_name'] = get_tc_names()
    print('# of normal distributions:', normal_dist_pairs)
    print('# of not normal distributions:', not_normal_dist_pairs)

    print('the heed of Results deltas labels')
    print(rd_df.head())

    return rd_df


def get_tc_names(start_step, end_step, epochs):
    tc_names = []
    for step in range(start_step, end_step):
        for tc_ind in range(0, epochs - step):
            for tc_ind2 in range(tc_ind + step, epochs - step):
                tc_names.append(
                    str(tc_ind) + '-' + str(tc_ind + step) + ':' + str(tc_ind2) + '-' + str(tc_ind2 + step))
    return tc_names


def init_rd_map(systems, metrics):
    res_change_map = {}
    for system in systems:
        for metric in metrics:
            res_change_map[system + '-' + metric] = []
    return res_change_map


def update_res_change_map(rd_dir, res_change_map, tc_ind1, tc_ind2, systems, metrics):
    rd_df = pd.read_csv(join(rd_dir, 'tc_' + str(tc_ind1) + '_' + str(tc_ind2) + '.csv'))
    for system in systems:
        for (i, metric) in enumerate(metrics):
            res_change_map[system + '-' + metric].append(rd_df[i][system])


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    # get the KD
    dtcs_content_dir = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.dtc_evolving_content_dir))
    vocab_dict = read_pickle(join(dtcs_content_dir, 'vocab.pkl'))

    tfidf_docs_kd_instance = tfidf_docs_kd.TIFIDFDocsKD(dtcs_content_dir, vocab_dict, cfg.dtc.n_evee,
                                                        cfg.dtc.start_step, cfg.dtc.end_step,
                                                        cfg.dtc.generate_vocab_stats)
    diff_vectors_l, diff_vectors_m, diff_vectors_u = tfidf_docs_kd_instance.calculate_kd()

    systems = [retrieval_model[index]['run_name'] for index, retrieval_model in enumerate(cfg.retrieval_models)]

    # get RD -- must execute first relative or absolute rd labelling
    rd_relative_mean_dir = join(cfg.config.root_dir,
                                join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evaluation.rd_relative_mean_path))
    rd_df = construct_rd_labels(rd_relative_mean_dir, cfg.dtc.start_step, cfg.dtc.end_step, cfg.dtc.n_evee)

    # train tfidf model
    kd_rd_svr.classify_cross_validation_tfidf(diff_vectors_l, diff_vectors_m, diff_vectors_u, vocab_dict, rd_df,
                                              join(cfg.config.root_dir,
                                                   cfg.kd_rd_prediction.svm.plot_data_prefix_tfidf),
                                              systems, cfg.evaluation_metrics)


if __name__ == '__main__':
    main()
