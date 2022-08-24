import string
from pprint import pprint

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from code.delta import clustering_interface
from code.preprocessing import normalizer_interface
from code.utils import io_util
from code.utils import preprocess_util


CUSTOM_STOP_WORDS = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
    'al.', 'Elsevier', 'PMC', 'CZI', 'www'
]


def read_files_to_df(input_path, files):
    dict_ = {clustering_interface.MAP_KEY__UID: [], clustering_interface.MAP_KEY__ABSTRACT: [],
             clustering_interface.MAP_KEY__BODY_TEXT: [], clustering_interface.MAP_KEY__TITLE: [],
             clustering_interface.MAP_KEY__ABSTRACT_SUMMARY: []}
    for idx, entry in enumerate(files):
        if idx % (len(files) // 10) == 0:
            print(f'Processing index: {idx} of {len(files)}')

        try:
            content = read_input_file(io_util.join(input_path, entry))
        except Exception as e:
            continue  # invalid paper format, skip

        dict_[clustering_interface.MAP_KEY__ABSTRACT].append(content[clustering_interface.MAP_KEY__ABSTRACT])
        dict_[clustering_interface.MAP_KEY__UID].append(content[clustering_interface.MAP_KEY__UID])
        dict_[clustering_interface.MAP_KEY__BODY_TEXT].append(content[clustering_interface.MAP_KEY__BODY_TEXT])
        dict_[clustering_interface.MAP_KEY__TITLE].append(content[clustering_interface.MAP_KEY__TITLE])
        dict_[clustering_interface.MAP_KEY__ABSTRACT_SUMMARY].append(
            construct_abstract_summary(content[clustering_interface.MAP_KEY__ABSTRACT]))

    df = pd.DataFrame(dict_, columns=[clustering_interface.MAP_KEY__UID, clustering_interface.MAP_KEY__ABSTRACT,
                                      clustering_interface.MAP_KEY__BODY_TEXT, clustering_interface.MAP_KEY__TITLE,
                                      clustering_interface.MAP_KEY__ABSTRACT_SUMMARY])
    return df


def add_text_fields_count(df):
    count_word_count_for_text_field(df, clustering_interface.MAP_KEY__ABSTRACT,
                                    clustering_interface.MAP_KEY__ABSTRACT_WORD_COUNT)
    count_word_count_for_text_field(df, clustering_interface.MAP_KEY__BODY_TEXT,
                                    clustering_interface.MAP_KEY__BODY_WORD_COUNT)
    count_word_count_for_text_field(df, clustering_interface.MAP_KEY__BODY_TEXT,
                                    clustering_interface.MAP_KEY__BODY_UNIQUE_WORD_COUNT, unique=True)


def clean_df(df):
    print("Removing duplicates ...")
    df.drop_duplicates([clustering_interface.MAP_KEY__ABSTRACT, clustering_interface.MAP_KEY__BODY_TEXT], inplace=True)
    df.info()

    print("Removing entries with empty values")
    df.dropna(inplace=True)
    df.info()

    print("Detecting languages ...")
    languages = detect_available_languages(df)
    df['language'] = languages
    print("Removing non English articles ..")
    df = df[df[clustering_interface.MAP_KEY__LANGUAGE] == 'en']
    df.info()


def process_body_text(df, stopwords, parser):
    print("processing body text ...")
    punctuations = string.punctuation

    tqdm.pandas()
    df[clustering_interface.MAP_KEY__PROCESSED_TEXT] = df["body_text"].progress_apply(preprocess_util.spacy_tokenizer,
                                                                                      args=(
                                                                                      parser, stopwords, punctuations))


def vectorize_processed_text(df, vectorizer):
    print("Vectorizing processed text ... ")
    text = df['processed_text'].values
    vectors = preprocess_util.vectorize_text(text, vectorizer)
    return vectors


def reduce_vectors(vectors, method):
    print("Reduce vectors ...")
    return method.fit_transform(vectors.toarray())


def generate_clusters(vectors, df, clustering_model):
    print("Generate K-means clusters ...")
    y_pred = clustering_model.fit_predict(vectors)
    df[clustering_interface.MAP_KEY__CLUSTER_LABEL] = y_pred


def plot_clusters(vectors, clusters_labels, title, output_path):
    # sns settings
    sns.set(rc={'figure.figsize': (15, 15)})

    # colors
    palette = sns.hls_palette(20, l=.4, s=.9)

    # plot
    sns.scatterplot(vectors[:, 0], vectors[:, 1], hue=clusters_labels, legend='full', palette=palette)
    plt.title(title)
    plt.savefig(output_path)
    plt.show()


def save_final_output(reduced_vectors, cluster_labels, df, output_dir):
    print("Save the final output to file ....")

    save_df_to_file(df, io_util.join(output_dir, "df_final.pkl"))

    # save the final t-SNE
    io_util.write_pickle(reduced_vectors, io_util.join(output_dir, "X_embedded.pkl"))

    # save the labels generate with k-means(20)
    io_util.write_pickle(cluster_labels, io_util.join(output_dir, "y_pred.pkl"))


def save_df_to_file(df, output_path):
    io_util.write_pickle(df, output_path)


def read_input_file(input_file_path):
    input_content = io_util.read_json(input_file_path)
    doc_data_map = {
        clustering_interface.MAP_KEY__UID: input_content[normalizer_interface.NormalizerInterface.MAP_KEY__UID],
        clustering_interface.MAP_KEY__TITLE: input_content[normalizer_interface.NormalizerInterface.MAP_KEY__METADATA]
        [normalizer_interface.NormalizerInterface.MAP_KEY__TITLE]
    }
    abstract_pars = []
    body_pars = []
    for par in input_content[normalizer_interface.NormalizerInterface.MAP_KEY__PARAGRAPHS]:
        if par[normalizer_interface.NormalizerInterface.MAP_KEY__SECTION][
            normalizer_interface.NormalizerInterface.MAP_KEY__TEXT].lower() == 'abstract':
            abstract_pars.append(par[normalizer_interface.NormalizerInterface.MAP_KEY__TEXT])
        else:
            body_pars.append(par[normalizer_interface.NormalizerInterface.MAP_KEY__TEXT])
    doc_data_map[clustering_interface.MAP_KEY__ABSTRACT] = '\n'.join(abstract_pars)
    doc_data_map[clustering_interface.MAP_KEY__BODY_TEXT] = '\n'.join(body_pars)
    return doc_data_map


def construct_abstract_summary(abstract):
    if len(abstract) == 0:
        # no abstract provided
        abstract_summary = "Not provided."
    elif len(abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 100 words append with ...
        info = abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        abstract_summary = summary + "..."
    else:
        # abstract is short enough
        abstract_summary = get_breaks(abstract, 40)
    return abstract_summary


def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data


def count_word_count_for_text_field(df, text_field_key, count_key, unique=False):
    if not unique:
        df[count_key] = df[text_field_key].apply(lambda x: len(x.strip().split()))
    else:
        df[count_key] = df[text_field_key].apply(lambda x: len(set(x.strip().split())))


def detect_available_languages(df):
    # hold label - language
    languages = []

    # go through each text
    for ii in tqdm(range(0, len(df))):
        # split by space into list, take the first x intex, join with space
        body_text = df.iloc[ii][clustering_interface.MAP_KEY__BODY_TEXT].split(" ")
        lang = preprocess_util.detect_text_language(df.iloc[ii][clustering_interface.MAP_KEY__BODY_TEXT].split(" "))
        if lang == "":
            lang = preprocess_util.detect_text_language(df.iloc[ii][clustering_interface.MAP_KEY__ABSTRACT].split(" "))
        languages.append(lang)

    languages_dict = {}
    for lang in set(languages):
        languages_dict[lang] = languages.count(lang)

    print("Total: {}\n".format(len(languages)))
    pprint(languages_dict)
    return languages


class NormalizedDataClustering(clustering_interface.ClusteringInterface):

    def __init__(self, language_model_parser, custom_stopwords, text_vectorizer_model, clustering_model):
        self.language_model_parser = language_model_parser
        self.custom_stopwords = custom_stopwords
        self.text_vectorizer_model = text_vectorizer_model
        self.clustering_model = clustering_model

    def build_clusters(self, input_path, output_path):
        if not io_util.path_exits(output_path):
            io_util.mkdir(output_path)

        if not io_util.path_exits(io_util.join(output_path, 'df_progress')):
            io_util.mkdir(io_util.join(output_path, 'df_progress'))

        normalized_docs = [file for file in io_util.list_files_in_dir(input_path) if file.endswith('.json')]
        df = read_files_to_df(input_path, normalized_docs)
        print(df.head())
        add_text_fields_count(df)
        print(df.head())
        clean_df(df)
        process_body_text(df, self.custom_stopwords, self.language_model_parser)
        save_df_to_file(df, io_util.join(output_path, 'df_progress/df_processed.pkl'))

        vectors = vectorize_processed_text(df, self.text_vectorizer_model)
        reduced_vectors_pca = reduce_vectors(vectors, PCA(n_components=0.95, random_state=42))
        generate_clusters(reduced_vectors_pca, df, self.clustering_model)
        reduced_vectors_tsne = reduce_vectors(vectors, TSNE(verbose=1, perplexity=100, random_state=42))

        if not io_util.path_exits(io_util.join(output_path, 'plots')):
            io_util.mkdir(io_util.join(output_path, 'plots'))

        plot_clusters(reduced_vectors_tsne, df[clustering_interface.MAP_KEY__CLUSTER_LABEL], 't-SNE with Kmeans Labels',
                      io_util.join(output_path, 'plots/improved_cluster_tsne.png'))

        if not io_util.path_exits(io_util.join(output_path, 'plot_data')):
            io_util.mkdir(io_util.join(output_path, 'plot_data'))
        save_final_output(reduced_vectors_tsne, df[clustering_interface.MAP_KEY__CLUSTER_LABEL], df,
                          io_util.join(output_path, 'plot_data'))
