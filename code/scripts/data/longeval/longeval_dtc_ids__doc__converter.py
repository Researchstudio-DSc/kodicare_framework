"""
Python script that takes the ids for created DTC
Then it extracts the title and the abstract from the content to json files that each represent a collection of documents
usage: python -m code.scripts.data.robust_dtc_ids_doc__converter --config-name robust_config config.root_dir='/root/path/to/collection'
"""
import hydra
import pandas as pd

from code.utils.io_util import *


def get_full_content(merged_collection_dir):
    docno_text_map = {}
    collectors = [file for file in list_files_in_dir(merged_collection_dir) if file.endswith('.json')]
    for collector in collectors:
        print(collector)
        collector_content = read_json(join(merged_collection_dir, collector))
        for index, info in enumerate(collector_content):
            docno_text_map[info['id']] = {'contents': info['contents']}
    return docno_text_map


def extract_contents_for_collection(full_content_map, doc_ids, out_dir, collection_id):
    doc_contents = []
    for index, doc_id in enumerate(doc_ids):
        doc_contents.append({
            'id': doc_id,
            'title': '',
            'contents': full_content_map[doc_id]['contents']
        })
    write_json(join(out_dir, 'tc_' + str(collection_id) + '.json'), doc_contents)


def extract_contents(dtc_ids_path, out_dir, docno_text_map):
    print("Reading the ids ....")
    dtc_ids = read_pickle(dtc_ids_path)

    for index, tc_id in enumerate(dtc_ids):
        print("Extracting the content for collection:", index)
        extract_contents_for_collection(docno_text_map, tc_id, out_dir, index)


def get_docno_list(csv_path):
    data = pd.read_csv(csv_path)
    return data['docno'].tolist()


@hydra.main(version_base=None, config_path="../../../../conf", config_name=None)
def main(cfg):
    merged_collection_dir = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.merged_collectors_dir))
    dtc_ids_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evee_info_path))
    output_dir = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.dtc_evolving_content_dir))

    print("Reading the full content ....")
    docno_text_map = get_full_content(merged_collection_dir)

    if not path_exits(output_dir):
        mkdir(output_dir)

    extract_contents(dtc_ids_path, output_dir, docno_text_map)


if __name__ == '__main__':
    main()
