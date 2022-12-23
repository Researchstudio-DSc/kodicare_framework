import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig

from code.indexing.index_util import Query
from code.utils import io_util



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
    return 1 / (10 + rank)


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
    query_reader = instantiate(cfg.evaluation.query_reader, data_dir=cfg.config.data_dir)
    queries = query_reader.read()

    ## INDEX
    if cfg.config.index_dir:
        index = instantiate(cfg.indexing.index, mode="load", index_dir=cfg.config.index_dir)
    else:
        index = instantiate(cfg.indexing.index, mode="load")

    # RANKING
    ranking_data = index.rank(queries=queries, size=cfg.evaluation.retrieval.size, query_builder=query_reader)

    run_file = io_util.join(cfg.config.out_dir, cfg.evaluation.retrieval.run_name)
    run_fp = open(run_file, "w")

    # TODO: support different output formats
    query_relevant_docs_map = []
    for query, query in zip(queries, ranking_data):
        q_id = query.id
        rank = 0
        ranking = []

        raw_ranking = query.relevant_docs
        # get general mapping to cord ids
        for entry in raw_ranking:
            document_id = entry[Query.KEY_SOURCE]['document_id']
            ranking.append((document_id, entry[Query.KEY_SCORE]))
        
        # rank aggregation
        if cfg.evaluation.retrieval.rank_aggregation:
            agg_ranking = call(cfg.evaluation.retrieval.rank_aggregation, ranking=ranking, agg_size=cfg.evaluation.retrieval.agg_size)
        else:
            agg_ranking = ranking
        
        for rank, (document_id, score) in enumerate(agg_ranking):
            run_fp.write(f"{q_id} Q0 {document_id} {rank} {score:.4f} {cfg.evaluation.retrieval.run_name}\n")
            ranked_relevant_docs.append({MAP_KEY__SCORE: score,
                                         MAP_KEY__INFO: {'document_id': document_id}})
        query_relevant_docs_map.append({MAP_KEY__QUERY_ID: str(q_i + 1), MAP_KEY__RELEVANT_DOCS: ranked_relevant_docs})
    run_fp.close()
    io_util.write_json(io_util.join(cfg.config.working_dir, cfg.evaluation.retrieval.run_name + '.json'),
                       query_relevant_docs_map)


if __name__ == "__main__":
    main()
