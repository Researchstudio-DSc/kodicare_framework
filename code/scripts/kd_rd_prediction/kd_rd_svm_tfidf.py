import hydra

from code.kd_rd_predition import kd_rd_svm
from code.knowledge_delta import tfidf_docs_kd
from code.utils.io_util import *


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

    # get RD -- must execute first binary_rd_labelling_executer
    rd_df = read_pickle(join(dtcs_content_dir, cfg.dtc.rd_label_path))

    # train tfidf model
    kd_rd_svm.classify_cross_validation_tfidf(diff_vectors_l, diff_vectors_m, diff_vectors_u, vocab_dict, rd_df,
                                              join(cfg.config.root_dir, cfg.kd_rd_prediction.svm.plot_data_prefix_tfidf),
                                              systems, cfg.evaluation_metrics)


if __name__ == '__main__':
    main()
