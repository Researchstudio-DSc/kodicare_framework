import hydra

from code.result_delta.binary_rd_labelling import *
from code.utils.io_util import *


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    dtcs_content_dir = join(cfg.config.root_dir, cfg.dtc.evaluation_splits_dir)
    evaluation_perquery_output_path = join(dtcs_content_dir, cfg.dtc.evaluation.evaluation_perquery_output_path)

    systems = [retrieval_model[index]['run_name'] for index, retrieval_model in enumerate(cfg.retrieval_models)]

    binary_rd_instance = BinaryRDLabelling(evaluation_perquery_output_path, cfg.dtc.epochs, cfg.dtc.start_step,
                                           cfg.dtc.end_step, systems, cfg.evaluation_metrics)
    rd_df = binary_rd_instance.construct_rd_labels()

    rd_df.to_pickle(join(dtcs_content_dir, cfg.dtc.rd_label_path))


if __name__ == '__main__':
    main()
