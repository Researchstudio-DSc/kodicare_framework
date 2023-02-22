"""
Script to generate the vector representation for documents collection using bag of words
1- vocab are collected for all documents
2- vector representation for each document by counting the vocab
3- vector representation is generated for a whole collection by summing up the documents vectors

example usage:
python -m code.scripts.representations.bow_doc_collection_representation --config-name cord19_config config.root_dir='/root/path/to/collection'
"""

import hydra
import string

from code.preprocessing import normalizer_interface
from code.representations import bow_representation
from code.utils import io_util
from code.utils import preprocess_util


def construct_docs_text(input_dir, docs):
    docs_ids = []
    docs_text = []
    for doc in docs:
        data = io_util.read_json(io_util.join(input_dir, doc))
        docs_ids.append(data[normalizer_interface.NormalizerInterface.MAP_KEY__DOC_ID])
        if len(data[normalizer_interface.NormalizerInterface.MAP_KEY__PARAGRAPHS]) > 0:
            docs_text.append(data[normalizer_interface.NormalizerInterface.MAP_KEY__PARAGRAPHS][0][
                                 normalizer_interface.NormalizerInterface.MAP_KEY__TEXT])
        else:
            docs_text.append("")
    return docs_ids, docs_text


def construct_vocab(docs_text, lang):
    vocab = []
    stopwords = preprocess_util.get_stopwords(language=preprocess_util.LANGUAGE_CODE_LANGUAGE__MAP[lang])
    for text in docs_text:
        tokens = preprocess_util.word_tokenize(text)
        tokens = preprocess_util.remove_stopwords(tokens, stopwords)
        tokens = preprocess_util.remove_punctuation(tokens, string.punctuation)

        vocab = vocab + list(set(tokens))
    return vocab


def represent_docs(doc_ids, docs_text, bow_instance, out_dir):
    docs_vectors = []
    for (index, text) in enumerate(docs_text):
        doc_id = doc_ids[index]
        print("Vectors for", doc_id)
        vector = bow_instance.represent_text(text)
        io_util.write_pickle(vector, io_util.join(out_dir, str(doc_id) + '_vec.pkl'))
        docs_vectors.append(vector)
    return docs_vectors


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    input_dir = io_util.join(cfg.config.root_dir,
                             io_util.join(cfg.config.working_dir, cfg.collection_normalization.out_dir))
    working_dir = io_util.join(cfg.config.root_dir,
                               io_util.join(cfg.config.working_dir, cfg.representation.bow_working_dir))
    out_prefix = cfg.representation.bow_output
    lang = cfg.config.lang

    if not io_util.path_exits(working_dir):
        io_util.mkdir(working_dir)

    docs = io_util.list_files_in_dir(input_dir)
    docs_ids, docs_text = construct_docs_text(input_dir, docs)
    print("1- Constructing the vocab of the documents")
    print("------------------------------------------")
    print()
    vocab = construct_vocab(docs_text, cfg.config.lang)

    bow_instance = bow_representation.BOWRepresentation(vocab, lang=lang)
    print("2- Generating vector representation for each document")
    print("-----------------------------------------------------")
    docs_vectors = represent_docs(docs_ids, docs_text, bow_instance, working_dir)
    print()
    print("3- Generating vector representation for each documents collection")
    print("-----------------------------------------------------------------")
    docs_representation = [sum(x) for x in zip(*docs_vectors)]
    io_util.write_text_to_file(io_util.join(working_dir, out_prefix + '_vec.txt'), str(docs_representation))
    io_util.write_text_to_file(io_util.join(working_dir, out_prefix + '_vocab.txt'), str(vocab))


if __name__ == '__main__':
    main()
