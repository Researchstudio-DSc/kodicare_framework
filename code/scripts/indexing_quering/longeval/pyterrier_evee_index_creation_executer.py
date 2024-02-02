"""
Python script that creates indices for the longeval simulated EvEE
usage: python -m code.scripts.indexing_quering.longeval.pyterrier_evee_longeval_index_creation_executer --config-name longeval_config config.root_dir='/root/path/to/collection'
"""
import hydra

from code.indexing.pyterrier_indexer import *


def get_text_loc_map(collectors_dirs):
    docno_text_loc_map = {}
    for collector_dir in collectors_dirs:
        print('Collecting info in dir', collector_dir)
        collectors = [file for file in list_files_in_dir(collector_dir) if file.endswith('.json')]
        for collector in collectors:
            print(collector)
            collector_content = read_json(join(collector_dir, collector))
            for index, info in enumerate(collector_content):
                docno_text_loc_map[info['id']] = (join(collector_dir, collector), index)
    return docno_text_loc_map


def index_evee(dtc_ids_path, index_path, docno_text_loc_map):
    print("Reading the ids ....")
    dtc_ids = read_pickle(dtc_ids_path)

    for index, tc_id in enumerate(dtc_ids):
        print("Indexing TC:", index)
        tc_name = 'tc_' + str(index)
        mkdir(join(index_path, tc_name))

        create_index_longeval_evee(join(index_path, tc_name), tc_id, docno_text_loc_map)


@hydra.main(version_base=None, config_path="../../../../conf", config_name=None)
def main(cfg):
    collectors_dirs = [join(cfg.config.root_dir, collectors_dir) for collectors_dir in cfg.dtc.collectors_dirs]
    dtc_ids_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evee_info_path))

    print("Reading the full content ....")
    docno_text_loc_map = get_text_loc_map(collectors_dirs)

    index_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.index.index_path))

    if not path_exits(index_path):
        mkdir(index_path)

    index_evee(dtc_ids_path, index_path, docno_text_loc_map)


if __name__ == '__main__':
    main()
