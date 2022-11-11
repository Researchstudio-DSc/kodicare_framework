"""
A python script to set the configuration and run the clustering for normalized cord 19 dataset
"""

import en_core_sci_lg
import hydra
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import \
    STOP_WORDS  # spacy and scispacy model usually cause conflict so the package version requirements must be set

from code.delta import normalized_data_clustering
from code.utils import io_util

CUSTOM_STOP_WORDS = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
    'al.', 'Elsevier', 'PMC', 'CZI', 'www'
]


@hydra.main(version_base=None, config_path="../../conf", config_name="cord19_config")
def main(cfg):
    input_dir = io_util.join(cfg.config.root_dir,
                             io_util.join(cfg.config.working_dir, cfg.collection_normalization.out_dir))
    output_dir = io_util.join(cfg.config.root_dir, io_util.join(cfg.config.working_dir, cfg.clustering.out_dir))

    if not io_util.path_exits(output_dir):
        io_util.mkdir(output_dir)

    stopwords = list(STOP_WORDS)
    for w in CUSTOM_STOP_WORDS:
        if w not in stopwords:
            stopwords.append(w)

    parser = en_core_sci_lg.load(disable=["tagger", "ner"])
    parser.max_length = 7000000

    vectorizer = TfidfVectorizer(max_features=2 ** 12)

    kmeans = KMeans(n_clusters=20, random_state=42)

    clustering_inst = normalized_data_clustering.NormalizedDataClustering(parser, stopwords, vectorizer, kmeans)
    clustering_inst.build_clusters(input_dir, output_dir)


if __name__ == '__main__':
    main()
