"""
Script to merge the longeval test collection (training and testing) into one collection after removing the duplicates
"""
import hydra
import pandas as pd

from code.utils.io_util import *


def get_merged_collection(collectors_dirs, docnos_set):
    merged_collection = []
    for collector_dir in collectors_dirs:
        print('Collecting info in dir', collector_dir)
        collectors = [file for file in list_files_in_dir(collector_dir) if file.endswith('.json')]
        for collector in collectors:
            print(collector)
            collector_content = read_json(join(collector_dir, collector))
            for index, info in enumerate(collector_content):
                if info['id'] not in docnos_set:
                    continue
                merged_collection.append(info)
    return merged_collection


def get_docno_list(csv_path):
    data = pd.read_csv(csv_path)
    return data['docno'].tolist()


def write_collectors(output_dir, merged_collection):
    start_index = 0
    end_index = 20000
    collector_index = 0
    while end_index != start_index:
        print('writing docs from', start_index, 'to', end_index, 'collector index', collector_index)
        write_json(join(output_dir, 'collector_' + str(collector_index) + '.json'),
                   merged_collection[start_index:end_index])
        start_index = end_index
        end_index = (end_index + 20000) if ((end_index + 20000) < len(merged_collection)) else len(merged_collection)
        collector_index += 1


@hydra.main(version_base=None, config_path="../../../../conf", config_name=None)
def main(cfg):
    collectors_dirs = [join(cfg.config.root_dir, collectors_dir) for collectors_dir in cfg.dtc.collectors_dirs]
    docno_date_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.docno_date_path))
    output_dir = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.merged_collectors_dir))

    print("Reading the full content ....")
    merged_collection = get_merged_collection(collectors_dirs, set(get_docno_list(docno_date_path)))

    if not path_exits(output_dir):
        mkdir(output_dir)

    write_collectors(output_dir, merged_collection)


if __name__ == '__main__':
    main()
