import pandas as pd
import pyterrier as pt

from code.utils.io_util import *

if not pt.started():
    pt.init(logging='TRACE', boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])


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


def create_index_json_longeval(index_path, docs_path, lang='en'):
    """
    create an index for the longeval json dataset
    :param lang: language of the test collection en or fr
    :param index_path: the path to store the index
    :param docs_path: the path of the json document collection
    :return: reference to the index
    """
    if not path_exits(join(index_path, 'data.properties')):
        print("Create nex index ..")
        iter_indexer = pt.IterDictIndexer(index_path, meta={'docno': 20, 'text': 4096})
        if lang == 'fr':
            iter_indexer = pt.IterDictIndexer(index_path, meta={'docno': 20, 'text': 4096},
                                              stemmer='FrenchSnowballStemmer', stopwords=None, tokeniser="UTFTokeniser")
        index_ref = iter_indexer.index(iter_dir_longeval(docs_path))
    else:
        print("Index already exists, an existing reference is returned ...")
        index_ref = pt.IndexRef.of(join(index_path, 'data.properties'))
    return index_ref


def iter_dir(docs):
    for doc in docs:
        doc['docno'] = doc.pop('id')
        doc['text'] = doc.pop('contents')
        yield doc


def create_index_longeval_evee(index_path, ee_docs_ids, docno_text_loc_map, lang='en'):
    """
    create an index for an EE from simulated evee for the longeval
    :param index_path: the path to store the index
    :param ee_docs_ids: list of docs ids in the ee
    :param docno_text_loc_map: a map of the location of each document in the full collection
    :return: reference to the index
    """
    if not path_exits(join(index_path, 'data.properties')):
        print("Create nex index ..")
        iter_indexer = pt.IterDictIndexer(index_path, meta={'docno': 20, 'text': 4096})
        if lang == 'fr':
            iter_indexer = pt.IterDictIndexer(index_path, meta={'docno': 20, 'text': 4096},
                                              stemmer='FrenchSnowballStemmer', stopwords=None, tokeniser="UTFTokeniser")
        index_ref = iter_indexer.index(iter_evee_docs_info(ee_docs_ids, docno_text_loc_map))
    else:
        print("Index already exists, an existing reference is returned ...")
        index_ref = pt.IndexRef.of(join(index_path, 'data.properties'))
    return index_ref


def iter_evee_docs_info(ee_docs_ids, docno_text_loc_map):
    for index, doc_id in enumerate(ee_docs_ids):
        print(index, doc_id)
        full_content_map = read_json(docno_text_loc_map[doc_id][0])
        doc = {
            'docno': full_content_map[docno_text_loc_map[doc_id][1]]['id'],
            'text': full_content_map[docno_text_loc_map[doc_id][1]]['contents']
        }
        yield doc


def create_index_json(index_path, docs_path):
    """
    create an index for the longeval json dataset
    :param index_path: the path to store the index
    :param docs_path: the json file of documents
    :return: reference to the index
    """
    if not path_exits(join(index_path, 'data.properties')):
        print("Create nex index ..")
        iter_indexer = pt.IterDictIndexer(index_path, meta={'docno': 20, 'text': 4096})
        index_ref = iter_indexer.index(iter_dir(read_json(docs_path)))
    else:
        print("Index already exists, an existing reference is returned ...")
        index_ref = pt.IndexRef.of(join(index_path, 'data.properties'))
    return index_ref


def read_queries_longeval(queries_path):
    queries_df = pd.read_csv(queries_path, sep='\t', names=['qid', 'query'])
    return queries_df


def read_queries_longeval_trec(queries_path):
    queries_df = pt.io.read_topics(queries_path)
    return queries_df


def read_qrels_longeval(qres_path):
    qrels_df = pd.read_csv(qres_path, sep="\s+", names=["qid", "rank", "docno", "label"])
    return qrels_df


def retrieve_run(index_path, queries_df, wmodel, controls={}, num_results=1000):
    """
    use pyterrier batch retrieval to return a df of results for retrieval
    :param index_path: the path of the created index by pyterrier
    :param queries_df: the queries dataframe
    :param wmodel: the scoring model e.g. TF-IDF, BM25
    (http://terrier.org/docs/current/javadoc/org/terrier/matching/models/package-summary.html)
    :param controls: parameters for query expansion
    (https://pyterrier.readthedocs.io/en/latest/terrier-retrieval.html#terrier-configuration)
    :param num_results: the number of rows for the results
    :return:
    """
    index_ref = pt.IndexRef.of(join(index_path, 'data.properties'))
    br_object = pt.BatchRetrieve(index_ref, wmodel=wmodel, controls=controls, num_results=num_results)
    run = br_object(queries_df)
    return run


def evaluate_run(run, run_name, qrels, metrics, perquery=False):
    """
    evaluate a run given a set of metrics and returns a data frame of each metric per query
    :param run: the data frame of the run result
    :param run_name: string that gives a run name
    :param qrels:
    :param metrics: list of evaluation metrics
    (http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system)
    :param perquery: if True then return the measure for each query otherwise the average result
    :return: dataframe of evaluation result
    """
    res_eval = pt.Utils.evaluate(run, qrels, metrics=metrics, perquery=perquery)
    print(res_eval)
    if not perquery:
        run_eval_df = pd.DataFrame({'metric': list(res_eval.keys()), run_name: list(res_eval.values())})
    else:
        run_eval_df = pd.DataFrame(res_eval).unstack().reset_index().rename(
            columns={'level_0': "qid", "level_1": "metric", 0: run_name})
    return run_eval_df


def evaluate_run_set(runs_dict, qrels, metrics, perquery=False):
    """
    evaluate a set of runs and returns a merged dataframe of all runs
    :param runs_dict: a dictionary of run name --> dataframe of retrieval
    :param qrels: the qrels
    :param metrics: list of evaluation metrics
    (http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system)
    :return: dataframe of evaluation result
    """
    evaluation_df = pd.DataFrame()
    for run in runs_dict:
        run_eval_df = evaluate_run(runs_dict[run], run, qrels, metrics, perquery=perquery)
        if len(evaluation_df) == 0:
            evaluation_df = run_eval_df
        else:
            evaluation_df = evaluation_df.merge(run_eval_df)
    return evaluation_df
