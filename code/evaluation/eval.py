import json
from lxml import etree
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from hydra.utils import instantiate, call

from code.indexing.index_util import Query

MAP_KEY__QUERY_TEXT = "query_text"
MAP_KEY__QUERY_ID = "query_id"
MAP_KEY__RELEVANT_DOCS = "relevant_docs"
MAP_KEY__DOC_ID = "doc_id"
MAP_KEY__UID = "uid"
MAP_KEY__SCORE = "score"
MAP_KEY__INFO = "info"

def agg_remove_duplicates(ranking, agg_size):
    # filter out duplicates, taking the first rank
    found_ids = set()
    reranking = []
    for cord_uid, score in ranking:
        if cord_uid in found_ids:
            continue
        found_ids.add(cord_uid)
        reranking.append((cord_uid, score))
    return reranking


def reciprocal_rank_score(rank):
    return 1/(10+rank)

def agg_paragraphs_recip(ranking, agg_size):
    # rerank by calculating a score based on the rank positions of a
    # document's paragraphs in the original ranking
    cord_uid_scores = {}
    for rank, (cord_uid, _) in enumerate(ranking):
        if cord_uid not in cord_uid_scores:
            cord_uid_scores[cord_uid] = 0
        cord_uid_scores[cord_uid] += reciprocal_rank_score(rank)
    return sorted(cord_uid_scores.items(), key=lambda x: x[1], reverse=True)[:agg_size]


@hydra.main(version_base=None, config_path="../../conf", config_name=None)
def main(cfg: DictConfig):
    ## QUERY READER
    queries_path = os.path.join(cfg.config.data_dir, cfg.evaluation.data.queries)
    query_reader = instantiate(cfg.evaluation.query_reader, queries_path=queries_path)
    queries = query_reader.read()

    ## INDEX
    index = instantiate(cfg.indexing.index, mode="load")

    ranking_data = index.rank(queries=queries, size=cfg.evaluation.retrieval.size, query_builder=query_reader)


    run_file = os.path.join(cfg.config.out_dir, cfg.evaluation.retrieval.run_name)
    run_fp = open(run_file, "w")
    ## RANKING
    # TODO: support different output formats
    for q_i, query in enumerate(ranking_data):
        rank = 0
        ranking = []

        raw_ranking = query.relevant_docs
        # get general mapping to cord ids
        for entry in raw_ranking:
            cord_uid = entry[Query.KEY_SOURCE]['cord_uid']
            ranking.append((cord_uid, entry[Query.KEY_SCORE]))
        
        # rank aggregation
        agg_ranking = call(cfg.evaluation.retrieval.rank_aggregation, ranking=ranking, agg_size=cfg.evaluation.retrieval.agg_size)
        
        for rank, (cord_uid, score) in enumerate(agg_ranking):
            run_fp.write(f"{q_i+1} Q0 {cord_uid} {rank} {score:.4f} {cfg.evaluation.retrieval.run_name}\n")
    run_fp.close()
    


if __name__ == "__main__":
    main()
