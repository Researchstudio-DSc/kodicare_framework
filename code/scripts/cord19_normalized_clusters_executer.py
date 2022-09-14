"""
A python script to set the configuration and run the clustering for normalized cord 19 dataset
"""

from code.delta import normalized_data_clustering

from spacy.lang.en.stop_words import \
    STOP_WORDS  # spacy and scispacy model usually cause conflict so the package version requirements must be set
import en_core_sci_lg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import argparse

CUSTOM_STOP_WORDS = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
    'al.', 'Elsevier', 'PMC', 'CZI', 'www'
]


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

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
    parser = argparse.ArgumentParser(
        description='Given a directory of json cord19 articles, normalize of the articles'
    )

    parser.add_argument('input_dir', help='Input directory of normalized data cord19 data')
    parser.add_argument('output_dir', help='The working directory for the progress and the final clusters')

    args = parser.parse_args()
    main(args)
