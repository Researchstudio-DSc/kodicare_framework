import pyterrier as pt

from code.utils.io_util import *

if not pt.started():
    pt.init(logging='TRACE')


def create_index_trec(index_path, docs_path):
    """
  create index for docs in the xml trec formate
  :param index_path: the path of the index to be stored in
  :param docs_path: the documents path in xml formate as specified by trec
  :param metadata: the metadata to be used during the index creation
  :return:
  """
    if not path_exits(join(index_path, 'data.properties')):
        docs_files = pt.io.find_files(docs_path)
        gen = pt.index.treccollection2textgen(docs_files)
        index = pt.IterDictIndexer(index_path).index(gen)
    else:
        index = pt.IndexRef.of(join(index_path, 'data.properties'))
    return index


def split_trec_file(input_path, output_path_prefix, max_docs=5000, file_extension='.txt'):
    lines = read_file_into_list(input_path)
    elements = []
    cur_docs = 0
    cur_split = 1
    for i in range(len(lines)):
        if lines[i].strip() == '<DOC>':
            cur_docs += 1
            while lines[i].strip() != '</DOC>':
                elements.append(lines[i])
                i += 1
            elements.append(lines[i])
            i += 1
        if cur_docs >= max_docs:
            write_list_to_file(output_path_prefix + '_' + str(cur_split) + file_extension, elements)
            elements = []
            cur_split += 1
            cur_docs = 0
    if len(elements) != 0:
        write_list_to_file(output_path_prefix + '_' + str(cur_split) + file_extension, elements)


def split_trec_document_collection(input_dir, output_dir, max_docs=5000, file_extension='.txt'):
    """
    split files of document collection written in trec format due to an exception which happen in indexing
    using pyterrier with big files
    :param input_dir: the directory of files for document collection
    :param output_dir: the output directory of the smaller chuncks of data
    :param max_docs: maximum number of documents per chunck
    :param file_extension: the extension of the files that contains the documents
    :return:
    """
    in_trec_files = [file for file in list_files_in_dir(input_dir) if file.endswith(file_extension)]
    if not path_exits(output_dir):
        mkdir(output_dir)
    for file in in_trec_files:
        split_trec_file(join(input_dir, file), join(output_dir, file[:-len(file_extension)]),
                        max_docs=max_docs, file_extension=file_extension)


def iter_dir_longeval(docs_path):
    files = [file for file in list_files_in_dir(docs_path) if file.endswith('.json')]
    for file in files:
        docs = read_json(join(docs_path, file))
        for doc in docs:
            doc['docno'] = doc.pop('id')
            doc['text'] = doc.pop('contents')
            yield doc


def create_index_json_longeval(index_path, docs_path):
    """
    create an index for the longeval json dataset
    :param index_path: the path to store the index
    :param docs_path: the path of the json document collection
    :return: reference to the index
    """
    if not path_exits(join(index_path, 'data.properties')):
        print("Create nex index ..")
        iter_indexer = pt.IterDictIndexer(index_path, meta={'docno': 20, 'text': 4096})
        index_ref = iter_indexer.index(iter_dir_longeval(docs_path))
    else:
        print("Index already exists, an existing reference is returned ...")
        index_ref = pt.IndexRef.of(join(index_path, 'data.properties'))
    return index_ref
