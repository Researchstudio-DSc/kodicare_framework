import os
import hydra
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../../conf", config_name=None)
def main(cfg):
    reader = instantiate(cfg.indexing.collection_reader, data_dir=cfg.config.data_dir)
    index = instantiate(cfg.indexing.index, mode="create")
    index.index_docs(reader.iterate())


if __name__ == '__main__':
    main()