"""
Python script that takes the ids for created DTC, a csv file of the full content
Then it extracts the title and the abstract from the content to json files that each represent a collection of documents
usage: python -m code.scripts.data.cord19_dtc_ids_doc__converter --config-name cord19_config config.root_dir='/root/path/to/collection'
"""
import hydra
import pandas as pd

from code.utils import io_util


def extract_contents_for_collection(df, doc_ids, out_dir, collection_id):
    mask = df[df['cord_uid'].isin(doc_ids.tolist())]
    doc_contents = []
    for index, row in mask.iterrows():
        doc_contents.append({
            'id': row['cord_uid'],
            'title': row['title'],
            'contents': row['abstract']
        })
    io_util.write_json(io_util.join(out_dir, 'tc_' + str(collection_id) + '.json'), doc_contents)


def extract_contents(dtc_ids_path, out_dir, full_contents_path):
    print("Reading the ids ....")
    dtc_ids = io_util.read_pickle(dtc_ids_path)
    print("Reading the full content ....")
    full_contents_df = pd.read_csv(full_contents_path, usecols=['cord_uid', 'title', 'abstract'])
    print(full_contents_df.head())

    full_contents_df.fillna('', inplace=True)
    for index, tc_id in enumerate(dtc_ids):
        print("Extracting the content for collection:", index)
        extract_contents_for_collection(full_contents_df, tc_id, out_dir, index)


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    dtc_ids_path = io_util.join(cfg.config.root_dir,
                                io_util.join(cfg.dtc.evaluation_splits_dir, cfg.dtc.dtc_evolving_ids_path))
    output_dir = io_util.join(cfg.config.root_dir, io_util.join(cfg.dtc.evaluation_splits_dir,
                                                                cfg.dtc.dtc_evolving_content_dir))
    full_contents_path = io_util.join(cfg.config.root_dir, io_util.join(cfg.dtc.evaluation_splits_dir,
                                                                        cfg.dtc.full_contents_path))

    if not io_util.path_exits(output_dir):
        io_util.mkdir(output_dir)

    extract_contents(dtc_ids_path, output_dir, full_contents_path)


if __name__ == '__main__':
    main()
