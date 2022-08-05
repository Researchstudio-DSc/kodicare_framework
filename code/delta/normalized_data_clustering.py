from code.delta import clustering_interface
from code.utils import io_util
from code.preprocessing import normalizer_interface
import pandas as pd
from tqdm import tqdm
from code.utils import preprocess_util
from pprint import pprint

MAP_KEY__UID = "uid"
MAP_KEY__ABSTRACT = "abstract"
MAP_KEY__BODY_TEXT = "body_text"
MAP_KEY__TITLE = "title"
MAP_KEY__ABSTRACT_SUMMARY = "abstract_summary"
MAP_KEY__ABSTRACT_WORD_COUNT = "abstract_word_count"
MAP_KEY__BODY_WORD_COUNT = "body_word_count"
MAP_KEY__BODY_UNIQUE_WORD_COUNT = "body_unique_word_count"
MAP_KEY__LANGUAGE = "language"


def read_files_to_df(input_path, files):
    dict_ = {MAP_KEY__UID: [], MAP_KEY__ABSTRACT: [], MAP_KEY__BODY_TEXT: [], MAP_KEY__TITLE: [],
             MAP_KEY__ABSTRACT_SUMMARY: []}
    for idx, entry in enumerate(files):
        if idx % (len(files) // 10) == 0:
            print(f'Processing index: {idx} of {len(files)}')

        try:
            content = read_input_file(io_util.join(input_path, entry))
        except Exception as e:
            continue  # invalid paper format, skip

        dict_[MAP_KEY__ABSTRACT].append(content[MAP_KEY__ABSTRACT])
        dict_[MAP_KEY__UID].append(content[MAP_KEY__UID])
        dict_[MAP_KEY__BODY_TEXT].append(content[MAP_KEY__BODY_TEXT])
        dict_[MAP_KEY__TITLE].append(content[MAP_KEY__TITLE])
        dict_[MAP_KEY__ABSTRACT_SUMMARY].append(construct_abstract_summary(content[MAP_KEY__ABSTRACT]))

    df = pd.DataFrame(dict_, columns=[MAP_KEY__UID, MAP_KEY__ABSTRACT, MAP_KEY__BODY_TEXT, MAP_KEY__TITLE,
                                      MAP_KEY__ABSTRACT_SUMMARY])
    return df


def add_text_fields_count(df):
    count_word_count_for_text_field(df, MAP_KEY__ABSTRACT, MAP_KEY__ABSTRACT_WORD_COUNT)
    count_word_count_for_text_field(df, MAP_KEY__BODY_TEXT, MAP_KEY__BODY_WORD_COUNT)
    count_word_count_for_text_field(df, MAP_KEY__BODY_TEXT, MAP_KEY__BODY_UNIQUE_WORD_COUNT, unique=True)


def clean_df(df):
    print("Removing duplicates ...")
    df.drop_duplicates([MAP_KEY__ABSTRACT, MAP_KEY__BODY_TEXT], inplace=True)
    df.info()

    print("Removing entries with empty values")
    df.dropna(inplace=True)
    df.info()

    print("Detecting languages ...")
    languages = detect_available_languages(df)
    df['language'] = languages
    print("Removing non English articles ..")
    df = df[df[MAP_KEY__LANGUAGE] == 'en']
    df.info()


def read_input_file(input_file_path):
    input_content = io_util.read_json(input_file_path)
    doc_data_map = {
        MAP_KEY__UID: input_content[normalizer_interface.NormalizerInterface.MAP_KEY__UID],
        MAP_KEY__TITLE: input_content[normalizer_interface.NormalizerInterface.MAP_KEY__METADATA]
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
    doc_data_map[MAP_KEY__ABSTRACT] = '\n'.join(abstract_pars)
    doc_data_map[MAP_KEY__BODY_TEXT] = '\n'.join(body_pars)
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
        body_text = df.iloc[ii][MAP_KEY__BODY_TEXT].split(" ")
        lang = preprocess_util.detect_text_language(df.iloc[ii][MAP_KEY__BODY_TEXT].split(" "))
        if lang == "":
            lang = preprocess_util.detect_text_language(df.iloc[ii][MAP_KEY__ABSTRACT].split(" "))
        languages.append(lang)

    languages_dict = {}
    for lang in set(languages):
        languages_dict[lang] = languages.count(lang)

    print("Total: {}\n".format(len(languages)))
    pprint(languages_dict)
    return languages


class NormalizedDataClustering(clustering_interface.ClusteringInterface):

    def build_clusters(self, input_path, output_path):
        normalized_docs = [file for file in io_util.list_files_in_dir(input_path) if file.endswith('.json')]
        df = read_files_to_df(input_path, normalized_docs)
        print(df.head())
        add_text_fields_count(df)
        print(df.head())
        clean_df(df)
        return
