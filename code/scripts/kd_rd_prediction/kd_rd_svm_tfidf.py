import hydra

from code.kd_rd_predition import kd_rd_svm
from code.knowledge_delta import tfidf_docs_kd
from code.utils import io_util


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    # get the KD
    dtcs_content_dir = io_util.join(cfg.config.root_dir, io_util.join(cfg.dtc.evaluation_splits_dir,
                                                                      cfg.dtc.dtc_evolving_content_dir))
    vocab_dict = io_util.read_pickle(io_util.join(dtcs_content_dir, 'vocab.pkl'))

    tfidf_docs_kd_instance = tfidf_docs_kd.TIFIDFDocsKD(dtcs_content_dir, vocab_dict, cfg.dtc.n_evee,
                                                        cfg.dtc.start_step, cfg.dtc.end_step,
                                                        cfg.dtc.generate_vocab_stats)
    diff_vectors_l, diff_vectors_m, diff_vectors_u = tfidf_docs_kd_instance.calculate_kd()

    # get RD -- must execute first binary_rd_labelling_executer
    rd_df = io_util.read_pickle(io_util.join(dtcs_content_dir, cfg.dtc.rd_label_path))

    # train tfidf model
    kd_rd_svm.classify_cross_validation_tfidf(diff_vectors_l, diff_vectors_m, diff_vectors_u, vocab_dict, rd_df,
                                              io_util.join(cfg.config.root_dir,
                                                           cfg.kd_rd_prediction.svm.plot_data_prefix_tfidf))


if __name__ == '__main__':
    main()
