"""
Python script that takes the ids for created DTC
Then it extracts the title and the abstract from the content to json files that each represent a collection of documents
usage: python -m code.scripts.data.robust_dtc_ids_doc__converter --config-name robust_config config.root_dir='/root/path/to/collection'
"""
import hydra
import ir_datasets

from code.utils import io_util


def get_full_content():
    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
    docno_text_map = {}
    for doc in dataset.docs_iter():
        docno_text_map[doc[0]] = {'title': doc[1].replace('\n', ' '), 'contents': doc[2].replace('\n', ' ')}
    return docno_text_map


def extract_contents_for_collection(full_content_map, doc_ids, out_dir, collection_id):
    doc_contents = []
    for index, doc_id in enumerate(doc_ids):
        doc_contents.append({
            'id': doc_id,
            'title': full_content_map[doc_id]['title'],
            'contents': full_content_map[doc_id]['contents']
        })
    io_util.write_json(io_util.join(out_dir, 'tc_' + str(collection_id) + '.json'), doc_contents)


def extract_contents(dtc_ids_path, out_dir, full_content_map):
    print("Reading the ids ....")
    dtc_ids = io_util.read_pickle(dtc_ids_path)

    for index, tc_id in enumerate(dtc_ids):
        print("Extracting the content for collection:", index)
        extract_contents_for_collection(full_content_map, tc_id, out_dir, index)


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    dtc_ids_path = io_util.join(cfg.config.root_dir,
                                io_util.join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evee_info_path))
    output_dir = io_util.join(cfg.config.root_dir, io_util.join(cfg.dtc.evaluation_splits_dir,
                                                                cfg.dtc.dtc_evolving_content_dir))

    print("Reading the full content ....")
    full_content_map = get_full_content()

    if not io_util.path_exits(output_dir):
        io_util.mkdir(output_dir)

    extract_contents(dtc_ids_path, output_dir, full_content_map)


if __name__ == '__main__':
    main()
