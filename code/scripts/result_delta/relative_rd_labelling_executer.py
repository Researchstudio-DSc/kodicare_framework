import hydra

from code.result_delta.relative_rd_labelling import RelativeRDLabelling
from code.utils.io_util import *


def generate_rd_for_dir(input_dir, output_dir, n_evee):
    files = sorted([file for file in list_files_in_dir(input_dir) if file.endswith('.csv')])
    for i in range(n_evee):
        for j in range(i + 1, n_evee):
            print(files[i], files[j])
            ee_results_path_1 = join(input_dir, files[i])
            ee_results_path_2 = join(input_dir, files[j])
            output_path = join(output_dir, 'tc_' + str(i) + '_' + str(j) + '.csv')
            relative_rd_inst = RelativeRDLabelling(ee_results_path_1, ee_results_path_2, output_path)
            relative_rd_inst.construct_rd_labels()


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    evaluation_perquery_dir = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir,
                                                             cfg.dtc.evaluation.evaluation_perquery_output_path))
    evaluation_mean_dir = join(cfg.config.root_dir,
                               join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evaluation.evaluation_mean_output_path))

    rd_relative_perquery_dir = join(cfg.config.root_dir,
                                    join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evaluation.rd_relative_perquery_path))
    rd_relative_mean_dir = join(cfg.config.root_dir,
                                join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evaluation.rd_relative_mean_path))

    if not path_exits(rd_relative_perquery_dir):
        mkdir(rd_relative_perquery_dir)
    if not path_exits(rd_relative_mean_dir):
        mkdir(rd_relative_mean_dir)

    generate_rd_for_dir(evaluation_perquery_dir, rd_relative_perquery_dir, cfg.dtc.n_evee)
    generate_rd_for_dir(evaluation_mean_dir, rd_relative_mean_dir, cfg.dtc.n_evee)


if __name__ == '__main__':
    main()
