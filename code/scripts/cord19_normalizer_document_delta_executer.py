"""
A python script to set the configuration and run the document delta caluclator for normalized cord 19 dataset
"""
from code.delta import normalized_document_delta_calculation
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse


def main(args):
    input_df_path = args.input_df_path
    output_path = args.output_path

    vectorizer = TfidfVectorizer(max_features=2 ** 12)

    cord19_delta_calculator = normalized_document_delta_calculation.\
        NormalizedDocumentDeltaCalculation(vectorizer, 'processed_text', 'pearson')
    cord19_delta_calculator.calculate_document_delta_score(input_df_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given a directory of json cord19 articles, normalize of the articles'
    )

    parser.add_argument('input_df_path', help='The data frame with documents clusters info')
    parser.add_argument('output_path', help='The output path of documents similarity')

    args = parser.parse_args()
    main(args)
