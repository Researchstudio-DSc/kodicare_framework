"""
A python script to set the configuration and run the document delta caluclator for normalized cord 19 dataset
"""
import hydra
from sklearn.feature_extraction.text import TfidfVectorizer

from code.delta import normalized_document_delta_calculation
from code.utils import io_util


@hydra.main(version_base=None, config_path="../../conf", config_name="cord19_config")
def main(cfg):
    input_df_path = io_util.join(io_util.join(cfg.config.root_dir,
                                              io_util.join(cfg.config.working_dir, cfg.clustering.out_dir)),
                                 "plot_data/df_final.pkl")
    output_path = io_util.join(cfg.config.root_dir, io_util.join(cfg.config.working_dir,
                                                                 cfg.delta.normalized_delta_path))

    vectorizer = TfidfVectorizer(max_features=2 ** 12)

    cord19_delta_calculator = normalized_document_delta_calculation. \
        NormalizedDocumentDeltaCalculation(vectorizer, 'processed_text', 'pearson')
    cord19_delta_calculator.calculate_document_delta_score(input_df_path, output_path)


if __name__ == '__main__':
    main()
