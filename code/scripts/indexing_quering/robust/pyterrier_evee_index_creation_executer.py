"""
python script to create index for longeval epoch using pyterrier
"""

import hydra

from code.indexing.pyterrier_indexer import *
from code.utils.io_util import *


@hydra.main(version_base=None, config_path="../../../../conf", config_name=None)
def main(cfg):
    evee_dir = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.dtc_evolving_content_dir))
    index_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.index.index_path))

    if not path_exits(index_path):
        mkdir(index_path)

    files = [file for file in list_files_in_dir(evee_dir) if file.endswith('.json')]

    for file in files:
        tc_name = file[:-5]
        mkdir(join(index_path, tc_name))
        create_index_json(join(index_path, tc_name), join(evee_dir, file))


if __name__ == '__main__':
    main()
