"""
Script to generate the vector representation for documents collection using tfidf
1- vocab are collected for all documents
2- vector representation for each document by counting the vocab
    a- generate the df matrix for the document
3- vector representation is generated for a whole collection by summing up the documents vectors

example usage:
python -m code.scripts.representations.tfidf_doc_collection_representation --config-name cord19_config config.root_dir='/root/path/to/collection'
"""

from multiprocessing import Queue, Process

import hydra
import numpy as np
import string

from code.preprocessing import normalizer_interface
from code.representations import tfidf_representation
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
    vocab = set()
    doc_text_tokens = []
    stopwords = preprocess_util.get_stopwords(language=preprocess_util.LANGUAGE_CODE_LANGUAGE__MAP[lang])
    for text in docs_text:
        tokens = preprocess_util.word_tokenize(text)
        tokens = preprocess_util.remove_stopwords(tokens, stopwords)
        tokens = preprocess_util.remove_punctuation(tokens, string.punctuation)

        vocab.update(tokens)
        doc_text_tokens.append(tokens)
    return list(vocab), doc_text_tokens


def construct_df(doc_text_tokens):
    df = {}
    for (index, doc_text) in enumerate(doc_text_tokens):
        for token in doc_text:
            if token not in df:
                df[token] = set()
            df[token].add(index)

    for token in df:
        df[token] = len(df[token])
    return df


def represent_docs(doc_ids, docs_text, tfidf_instance, out_dir):
    docs_id_queue = Queue()
    for doc_id in doc_ids:
        docs_id_queue.put(doc_id)
    docs_text_queue = Queue()
    for text in docs_text:
        docs_text_queue.put(text)

    procs = [Process(target=represent_docs_worker, args=[i, docs_id_queue, docs_text_queue, tfidf_instance, out_dir])
             for i in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def represent_docs_worker(proc_num, docs_id_queue, docs_text_queue, bow_instance, out_dir):
    while True:
        if docs_id_queue.empty():
            break
        doc_id = docs_id_queue.get()
        text = docs_text_queue.get()
        print(proc_num, "Vectors for", doc_id)
        vector = bow_instance.represent_text(text)
        io_util.write_pickle(np.array(vector), io_util.join(out_dir, str(doc_id) + '_vec.pkl'))


def represent_doc_collection(doc_ids, vector_len, out_dir):
    vector = np.zeros(vector_len)
    for doc_id in doc_ids:
        vector = np.add(vector, io_util.read_pickle(io_util.join(out_dir, str(doc_id) + '_vec.pkl')))
    return vector


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    input_dir = io_util.join(cfg.config.root_dir,
                             io_util.join(cfg.config.working_dir, cfg.collection_normalization.out_dir))
    working_dir = io_util.join(cfg.config.root_dir,
                               io_util.join(cfg.config.working_dir, cfg.representation.tfidf_working_dir))
    out_prefix = cfg.representation.tfidf_output
    lang = cfg.config.lang

    if not io_util.path_exits(working_dir):
        io_util.mkdir(working_dir)

    docs = io_util.list_files_in_dir(input_dir)
    docs_ids, docs_text = construct_docs_text(input_dir, docs)
    print("1- Constructing the vocab of the documents")
    print("------------------------------------------")
    print()
    vocab, doc_text_tokens = construct_vocab(docs_text, cfg.config.lang)

    print("2- Generating vector representation for each document")
    print("-----------------------------------------------------")
    print("a- Construct the DF of a document.")
    df = construct_df(doc_text_tokens)
    tfidf_instance = tfidf_representation.TFIDFRepresentation(vocab, df, len(docs_text), lang=lang)
    represent_docs(docs_ids, docs_text, tfidf_instance, working_dir)
    print()
    print("3- Generating vector representation for each documents collection")
    print("-----------------------------------------------------------------")
    docs_representation = represent_doc_collection(docs_ids, len(vocab), working_dir)
    io_util.write_pickle(docs_representation, io_util.join(working_dir, out_prefix + '_vec.pkl'))
    io_util.write_text_to_file(io_util.join(working_dir, out_prefix + '_vocab.txt'), str(vocab))


if __name__ == '__main__':
    main()
