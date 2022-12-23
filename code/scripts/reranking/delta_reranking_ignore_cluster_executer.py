"""
A python script to set the configuration and run the reranking for the delta clustering
"""

import hydra

from code.retrieval.reranking import delta_reranking_clusters_ignored
from code.utils import io_util


@hydra.main(version_base=None, config_path="../../conf", config_name="cord19_config")
def main(cfg):
    doc_clusters_path = io_util.join(io_util.join(cfg.config.root_dir,
                                                  io_util.join(cfg.config.working_dir, cfg.clustering.out_dir)),
                                     "plot_data/df_final.pkl")
    doc_pairs_similarity_path = io_util.join(cfg.config.root_dir, io_util.join(cfg.config.working_dir,
                                                                               cfg.delta.normalized_delta_path))
    base_retrieval_output_path = io_util.join(cfg.config.root_dir, io_util.join(cfg.config.working_dir,
                                                                                cfg.delta.base_retrieval_path))
    reranked_retrieval_output_path = io_util.join(cfg.config.root_dir, io_util.join(cfg.config.output_dir,
                                                                                    cfg.retrieval.rerank_path))

    rerank_cluster_ignored = delta_reranking_clusters_ignored.DeltaRerankingClustersIgnored(doc_clusters_path,
                                                                                            doc_pairs_similarity_path)

    rerank_cluster_ignored.rerank_retrieval_output(base_retrieval_output_path, reranked_retrieval_output_path)


if __name__ == '__main__':
    main()
