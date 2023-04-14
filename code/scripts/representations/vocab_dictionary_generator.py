"""
Python script to construct a vocab dictionary of documents of test collection represented by a group of json files
"""

import gensim
import hydra
import pandas as pd
import string

from code.utils import io_util
from code.utils import preprocess_util


def preprocess(text):
    return preprocess_util.execute_common_preprocess_pipeline(text, string.punctuation)


def construct_vocab(collections, dtcs_content_dir):
    df = pd.DataFrame()
    for collection in collections:
        df = df.append(pd.read_json(io_util.join(dtcs_content_dir, collection)))
        df.drop_duplicates(inplace=True)
    return df


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    dtcs_content_dir = io_util.join(cfg.config.root_dir, io_util.join(cfg.dtc.evaluation_splits_dir,
                                                                      cfg.dtc.dtc_evolving_content_dir))
    print(dtcs_content_dir)
    collections = [file for file in io_util.list_files_in_dir(dtcs_content_dir) if file.endswith('json')]
    collections_df = construct_vocab(collections, dtcs_content_dir)

    collections_df['merged_text'] = collections_df['title'] + ' ' + collections_df['contents']
    print(collections_df.head())

    processed_texts = collections_df['merged_text'].map(preprocess)

    print(processed_texts[:10])

    dictionary = gensim.corpora.Dictionary(processed_texts)

    print(len(dictionary))

    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    io_util.write_pickle(dictionary, io_util.join(dtcs_content_dir, 'vocab.pkl'))


if __name__ == '__main__':
    main()
