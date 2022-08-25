"""
A python script to set the configuration and run the reranking for the delta clustering
"""

from code.retrieval.reranking import delta_reranking_clusters_first
import argparse


def main(args):
    doc_clusters_path = args.doc_clusters_path
    doc_pairs_similarity_path = args.doc_pairs_similarity_path
    base_retrieval_output_path = args.base_retrieval_output_path
    reranked_retrieval_output_path = args.reranked_retrieval_output_path

    rerank_cluster_first = delta_reranking_clusters_first.DeltaRerankingClustersFirst(doc_clusters_path,
                                                                                      doc_pairs_similarity_path)

    rerank_cluster_first.rerank_retrieval_output(base_retrieval_output_path, reranked_retrieval_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to run the reranking of clustering first'
    )

    parser.add_argument('doc_clusters_path', help='The pickle file of df for the clusters')
    parser.add_argument('doc_pairs_similarity_path', help='The doc pairs similarity json path')
    parser.add_argument('base_retrieval_output_path', help='The json path of the baseline retrieval')
    parser.add_argument('reranked_retrieval_output_path', help='The json path of the results of reranking')

    args = parser.parse_args()
    main(args)
