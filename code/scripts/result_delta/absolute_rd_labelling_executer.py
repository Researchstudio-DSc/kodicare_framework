import hydra

from code.result_delta.absolute_rd_labelling import AbsoluteRDLabelling
from code.utils.io_util import *


def generate_rd_for_dir(input_dir, output_dir):
    files = sorted([file for file in list_files_in_dir(input_dir) if file.endswith('.csv')])
    for i in range(len(files) - 1):
        for j in range(i + 1, len(files)):
            print(files[i], files[j])
            ee_results_path_1 = join(input_dir, files[i])
            ee_results_path_2 = join(input_dir, files[j])
            output_path = join(output_dir, 'tc_' + files[i][3:-4] + '_' + files[j][3:-4] + '.csv')
            absolute_rd_inst = AbsoluteRDLabelling(ee_results_path_1, ee_results_path_2, output_path)
            absolute_rd_inst.construct_rd_labels()


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    evaluation_perquery_dir = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir,
                                                             cfg.dtc.evaluation.evaluation_perquery_output_path))
    evaluation_mean_dir = join(cfg.config.root_dir,
                               join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evaluation.evaluation_mean_output_path))

    rd_absolute_perquery_dir = join(cfg.config.root_dir,
                                    join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evaluation.rd_absolute_perquery_path))
    rd_absolute_mean_dir = join(cfg.config.root_dir,
                                join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evaluation.rd_absolute_mean_path))

    if not path_exits(rd_absolute_perquery_dir):
        mkdir(rd_absolute_perquery_dir)
    if not path_exits(rd_absolute_mean_dir):
        mkdir(rd_absolute_mean_dir)

    generate_rd_for_dir(evaluation_perquery_dir, rd_absolute_perquery_dir)
    generate_rd_for_dir(evaluation_mean_dir, rd_absolute_mean_dir)


if __name__ == '__main__':
    main()
