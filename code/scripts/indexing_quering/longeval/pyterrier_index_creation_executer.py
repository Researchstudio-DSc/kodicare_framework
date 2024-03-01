"""
python script to create index for longeval epoch using pyterrier
"""

from code.indexing.pyterrier_indexer import *
from code.utils.io_util import *
import hydra


@hydra.main(version_base=None, config_path="../../../../conf", config_name=None)
def main(cfg):
    documents_dir = join(cfg.config.root_dir, cfg.test_collection.documents_dir)
    index_path = join(cfg.config.root_dir, cfg.index.index_path)

    if not path_exits(index_path):
        mkdir(index_path)

    create_index_json_longeval(index_path, documents_dir)


if __name__ == '__main__':
    main()
