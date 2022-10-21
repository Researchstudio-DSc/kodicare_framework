import json
from lxml import etree
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from hydra.utils import instantiate

from code.indexing.es_index import Index
from code.utils.cord19_reader import CORD19Reader, CORD19ParagraphReader

MAP_KEY__QUERY_TEXT = "query_text"
MAP_KEY__QUERY_ID = "query_id"
MAP_KEY__RELEVANT_DOCS = "relevant_docs"
MAP_KEY__DOC_ID = "doc_id"
MAP_KEY__UID = "uid"
MAP_KEY__SCORE = "score"
MAP_KEY__INFO = "info"

def doc_reranking(cord_ranking):
    # rerank the documents using the original ranking
    # but filter out duplicates
    found_ids = set()
    reranking = []
    for cord_uid, score in cord_ranking:
        if cord_uid in found_ids:
            continue
        found_ids.add(cord_uid)
        reranking.append((cord_uid, score))
    return reranking


def reciprocal_rank_score(rank):
    return 1/(10+rank)

def paragraph_reranking(cord_ranking):
    # rerank by calculating a score based on the rank positions of a
    # document's paragraphs in the original ranking
    cord_uid_scores = {}
    for rank, (cord_uid, _) in enumerate(cord_ranking):
        if cord_uid not in cord_uid_scores:
            cord_uid_scores[cord_uid] = 0
        cord_uid_scores[cord_uid] += reciprocal_rank_score(rank)
    return sorted(cord_uid_scores.items(), key=lambda x: x[1], reverse=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(cfg : DictConfig):
    ## QUERY READER
    queries_path = os.path.join(cfg.config.data_dir, cfg.evaluation.data.queries)
    query_reader = instantiate(cfg.evaluation.query_reader, queries_path=queries_path)
    queries = query_reader.read()

    ## MOVE TO INDEX
    
    with open(os.path.join(cfg.config.data_dir, cfg.evaluation.data.cord_id_title), "r") as fp:
        cord_id_title = json.load(fp)
    
    # mapping between IDs used in qrel_file and files in index
    cord_uid_mapping = {}
    for mapping in cord_id_title:
        # paper_id field can contain multiple ids separated by ';'
        paper_ids = [p.strip() for p in mapping['paper_id'].split(';')]
        for paper_id in paper_ids:
            cord_uid_mapping[paper_id] = mapping['cord_uid']

    ## DOCUMENT READER
    if args.index_type == "paragraphs":
        reader = CORD19ParagraphReader(batch_size=16384)
    else:
        reader = CORD19Reader(batch_size=1024)
    index = Index(args.index_name, host=args.host)

    ranking_data = index.rank(queries=queries, size=args.size, query_builder=reader)


    for q_i, ranking_obj in enumerate(ranking_data):
        rank = 0
        cord_ranking = []

        ranking = ranking_obj[MAP_KEY__RELEVANT_DOCS]
        # get general mapping to cord ids
        for entry in ranking:
            paper_id = entry[MAP_KEY__INFO]['doc_id']
            cord_uid = cord_uid_mapping[paper_id]
            cord_ranking.append((cord_uid, entry[MAP_KEY__SCORE]))
        
        if args.index_type == "paragraphs":
            reranking = paragraph_reranking(cord_ranking)
        else: # whole documents
            reranking = doc_reranking(cord_ranking)
        
        for rank, (cord_uid, score) in enumerate(reranking):
            print(f"{q_i+1} Q0 {cord_uid} {rank} {score:.4f} {args.run_name}")
    

        




if __name__ == "__main__":
    main()