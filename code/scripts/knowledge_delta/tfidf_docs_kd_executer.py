import hydra

from code.knowledge_delta import tfidf_docs_kd
from code.utils import io_util


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    dtcs_content_dir = io_util.join(cfg.config.root_dir, io_util.join(cfg.dtc.evaluation_splits_dir,
                                                                      cfg.dtc.dtc_evolving_content_dir))
    vocab_dict = io_util.read_pickle(io_util.join(dtcs_content_dir, 'vocab.pkl'))

    tfidf_docs_kd_instance = tfidf_docs_kd.TIFIDFDocsKD(dtcs_content_dir, vocab_dict, cfg.dtc.n_evee,
                                                        cfg.dtc.start_step, cfg.dtc.end_step,
                                                        cfg.dtc.generate_vocab_stats)
    tfidf_docs_kd_instance.calculate_kd()


if __name__ == '__main__':
    main()
