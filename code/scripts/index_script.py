import os
import hydra
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../../conf", config_name=None)
def main(cfg):
    collection_path = os.path.join(cfg.config.data_dir, cfg.indexing.collection)
    reader = instantiate(cfg.indexing.collection_reader, collection_path=collection_path)
    index = instantiate(cfg.indexing.index, mode="create")
    index.index_docs(reader.iterate())


if __name__ == '__main__':
    main()