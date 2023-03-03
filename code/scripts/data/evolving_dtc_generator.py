"""
python script that takes hdf5 file as an input and generates the dtc
output: dtcs in the form of 2d array of docs ids
"""
import hydra
import numpy as np
import pandas as pd

from code.data import evolving_dtc_splits_parser
from code.utils import io_util


def generate_cont_evolving_dtc(evolving_dtc_parser):
    dtc_evolving_cont = evolving_dtc_parser.create_overlap_sample(random_init=False, rm_rate=0)
    dtc_evolving_cont = dtc_evolving_cont[list(dtc_evolving_cont.keys())[0]]['doc']
    return dtc_evolving_cont


def generate_random_evolving_dtc(evolving_dtc_parser):
    dtc_evolving_cont = evolving_dtc_parser.create_overlap_sample(random_init=False, rm_rate=1)
    dtc_evolving_cont = dtc_evolving_cont[list(dtc_evolving_cont.keys())[0]]['doc']
    return dtc_evolving_cont


def generate_random_dtc(evolving_dtc_parser):
    dtc_evolving_cont = evolving_dtc_parser.create_overlap_sample(tc_type='random', random_init=False, rm_rate=1)
    dtc_evolving_cont = dtc_evolving_cont[list(dtc_evolving_cont.keys())[0]]['doc']
    return dtc_evolving_cont


def convert_dtc_id_to_original_id(csv_info_path, evolving_dtc_ids):
    df = pd.read_csv(csv_info_path)
    evolving_dtc_orig_ids = []

    for index, tc_ids in enumerate(evolving_dtc_ids):
        mask = df.iloc[tc_ids]
        print(len(tc_ids), len(mask['docno'].tolist()))
        evolving_dtc_orig_ids.append(np.array(mask['docno'].tolist()))
    return evolving_dtc_orig_ids


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    evaluation_splits_path = io_util.join(cfg.config.root_dir,
                                          io_util.join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evaluation_splits_path))
    evolving_dtc_parser = evolving_dtc_splits_parser.EvolvingDTCSplitsParser(evaluation_splits_path)
    docno_date_path = io_util.join(cfg.config.root_dir,
                                   io_util.join(cfg.dtc.evaluation_splits_dir, cfg.dtc.docno_date_path))

    print("getting the evolving test collection without remove")
    dtc_evolving_cont = generate_cont_evolving_dtc(evolving_dtc_parser)
    dtc_orig = convert_dtc_id_to_original_id(docno_date_path, dtc_evolving_cont)
    io_util.write_pickle(dtc_orig,
                         io_util.join(cfg.config.root_dir,
                                      io_util.join(cfg.dtc.evaluation_splits_dir, cfg.dtc.dtc_evolving_cont_ids_path)))

    print("getting the evolving test collection with random remove")
    # date is considered in the order of the test collection
    dtc_evolving = generate_random_evolving_dtc(evolving_dtc_parser)
    dtc_orig = convert_dtc_id_to_original_id(docno_date_path, dtc_evolving)
    io_util.write_pickle(dtc_orig,
                         io_util.join(cfg.config.root_dir,
                                      io_util.join(cfg.dtc.evaluation_splits_dir, cfg.dtc.dtc_evolving_ids_path)))

    print("getting random test collections")
    # date is considered in the order of the test collection
    dtc_random = generate_random_dtc(evolving_dtc_parser)
    dtc_orig = convert_dtc_id_to_original_id(docno_date_path, dtc_random)
    io_util.write_pickle(dtc_orig,
                         io_util.join(cfg.config.root_dir,
                                      io_util.join(cfg.dtc.evaluation_splits_dir, cfg.dtc.dtc_random_ids_path)))


if __name__ == '__main__':
    main()
