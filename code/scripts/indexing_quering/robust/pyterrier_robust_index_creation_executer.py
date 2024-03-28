"""
python script to create index for the whole robust test collection
https://pyterrier.readthedocs.io/en/latest/experiments/Robust04.html
"""

import hydra

from code.indexing.pyterrier_indexer import *


@hydra.main(version_base=None, config_path="../../../../conf", config_name=None)
def main(cfg):
    documents_dir = join(cfg.config.root_dir, cfg.test_collection.documents_dir)
    index_path = join(cfg.config.root_dir, cfg.index.index_path)

    if not path_exits(index_path):
        mkdir(index_path)

    files = pt.io.find_files(documents_dir)
    # no-one indexes the congressional record in directory /CR/
    # indeed, recent copies from NIST dont contain it
    # we also remove some of the other unneeded files
    bad = ['/CR/', '/AUX/', 'READCHG', 'READFRCG']
    for b in bad:
        files = list(filter(lambda f: b not in f, files))
    indexer = pt.TRECCollectionIndexer(index_path, verbose=True)
    indexer.index(files)


if __name__ == '__main__':
    main()
