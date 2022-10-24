import os
import hydra
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../../conf", config_name=None)
def main(cfg):

    reader = instantiate(cfg.indexing.collection_reader)
    index = instantiate(cfg.indexing.index, mode="create")

    document_full_dir = os.path.join(cfg.config.data_dir, cfg.indexing.document_folder)
    index.index_docs(reader.iterate(document_full_dir))


if __name__ == '__main__':
    main()