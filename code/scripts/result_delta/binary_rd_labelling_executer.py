import hydra

from code.result_delta import binary_rd_labelling
from code.utils import io_util


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    dtcs_content_dir = io_util.join(cfg.config.root_dir, cfg.dtc.evaluation_splits_dir)

    evaluation_file = cfg.dtc.evaluation_splits_path

    binary_rd_instance = binary_rd_labelling.BinaryRDLabelling(dtcs_content_dir, evaluation_file, cfg.config.epochs,
                                                               cfg.config.start_step, cfg.config.end_step, )
    rd_df = binary_rd_instance.construct_rd_labels()

    rd_df.to_pickle(io_util.join(dtcs_content_dir, cfg.dtc.rd_label_path))


if __name__ == '__main__':
    main()
