"""
python script to filter a csv file contains information about date and original document id to get these info only
in another csv with the aim to be used in create of DTC
usage: python -m code.scripts.data.convert_to_docno_date --config-name cord19_config config.root_dir='/root/path/to/collection'
"""
import hydra
import pandas as pd

from code.utils import io_util


def filter_docno_date(input_path, out_path, out_in_fields_map):
    df = pd.read_csv(input_path, usecols=list(out_in_fields_map.values()))
    print(df.head())
    df.columns = list(out_in_fields_map.keys())
    print(df.head())
    df.to_csv(out_path)
    return df


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    input_path = io_util.join(cfg.config.root_dir, io_util.join(cfg.dtc.evaluation_splits_dir,
                                                                cfg.dtc.docs_full_info_path))
    output_path = io_util.join(cfg.config.root_dir, io_util.join(cfg.dtc.evaluation_splits_dir,
                                                                 cfg.dtc.docno_date_path))
    filter_docno_date(input_path, output_path, cfg.dtc.out_in_fields_map)


if __name__ == '__main__':
    main()
