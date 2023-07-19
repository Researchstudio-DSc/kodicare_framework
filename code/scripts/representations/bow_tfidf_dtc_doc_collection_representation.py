import hydra

from code.representations import tfidf_document_collection_representation
from code.utils import io_util


def represent_doc_collection(dtcs_content_dir, collection_path, out_dir, collection_id, bow_instance, vocab_dict):
    print("Representing collection:", collection_id)

    tfidf_doc_collection_rep_instance = tfidf_document_collection_representation.TFIDFDocCollectionRepresentation(
        vocab_dict, io_util.join(dtcs_content_dir, collection_path))

    merged_bow_vec, merged_tfidf_vec = tfidf_doc_collection_rep_instance.represent_document_collection()

    io_util.write_pickle(merged_bow_vec, io_util.join(out_dir, str(collection_id) + '_bow_vec.pkl'))
    io_util.write_pickle(merged_tfidf_vec, io_util.join(out_dir, str(collection_id) + '_tfidf_vec.pkl'))
    io_util.write_pickle(bow_vectors, io_util.join(out_dir, str(collection_id) + '_bow_full_vec.pkl'))
    io_util.write_pickle(tfidf_vectors, io_util.join(out_dir, str(collection_id) + '_tfidf_full_vec.pkl'))
    tfidf.save(io_util.join(out_dir, str(collection_id) + '_tfidf_model.pkl'))


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    dtcs_content_dir = io_util.join(cfg.config.root_dir, io_util.join(cfg.dtc.evaluation_splits_dir,
                                                                      cfg.dtc.dtc_evolving_content_dir))

    collections = [file for file in io_util.list_files_in_dir(dtcs_content_dir) if file.endswith('json')]

    vocab_dict = io_util.read_pickle(io_util.join(dtcs_content_dir, 'vocab.pkl'))

    for collection in collections:
        represent_doc_collection(dtcs_content_dir, collection, dtcs_content_dir, collection[3:-5], vocab_dict)


if __name__ == '__main__':
    main()
