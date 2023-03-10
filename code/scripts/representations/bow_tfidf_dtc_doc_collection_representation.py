import hydra
import pandas as pd
from gensim import models

from code.representations import bow_representation
from code.utils import io_util


def represent_doc_collection(dtcs_content_dir, collection_path, out_dir, collection_id, bow_instance, vocab_dict):
    print("Representing collection:", collection_id)

    df = pd.read_json(io_util.join(dtcs_content_dir, collection_path))
    df['merged_text'] = df['title'] + ' ' + df['contents']

    bow_vectors = [bow_instance.represent_text(doc) for doc in df['merged_text']]
    print(len(bow_vectors))
    tfidf = models.TfidfModel(bow_vectors)
    tfidf_vectors = tfidf[bow_vectors]

    # sum the vectors
    bow_dict = {}
    tfidf_dict = {}
    for k, v in vocab_dict.iteritems():
        bow_dict[k] = tfidf_dict[k] = 0

    for vec in bow_vectors:
        for name, num in vec:
            bow_dict[name] += num
    for vec in tfidf_vectors:
        for name, num in vec:
            tfidf_dict[name] += num

    # using map
    merged_bow_vec = list(map(tuple, bow_dict.items()))
    merged_tfidf_vec = list(map(tuple, tfidf_dict.items()))
    io_util.write_pickle(merged_bow_vec, io_util.join(out_dir, str(collection_id) + '_bow_vec.pkl'))
    io_util.write_pickle(merged_tfidf_vec, io_util.join(out_dir, str(collection_id) + '_tfidf_vec.pkl'))


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    dtcs_content_dir = io_util.join(cfg.config.root_dir, io_util.join(cfg.dtc.evaluation_splits_dir,
                                                                      cfg.dtc.dtc_evolving_content_dir))

    collections = [file for file in io_util.list_files_in_dir(dtcs_content_dir) if file.endswith('json')]

    vocab_dict = io_util.read_pickle(io_util.join(dtcs_content_dir, 'vocab.pkl'))
    print(vocab_dict)

    bow_instance = bow_representation.BOWRepresentation(vocab_dict, lang=cfg.config.lang)

    for collection in collections:
        represent_doc_collection(dtcs_content_dir, collection, dtcs_content_dir, collection[3:-5], bow_instance,
                                 vocab_dict)


if __name__ == '__main__':
    main()
