from quant_pre_utils import *

from top2vec import Top2Vec
import re
import re
import nltk
from nltk.corpus import stopwords


def read_json_file(file_path: str) -> List[dict]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def json_files_generator(folder_path: str) -> List[str]:
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            yield os.path.join(folder_path, file)


def extract_contents(json_data: List[dict]) -> List[str]:
    contents = [item['contents'] for item in json_data]
    return contents


def random_sample_documents(documents: List[str], sample_ratio: float = 0.33) -> List[str]:
    print("Sampling documents")
    sample_size = int(len(documents) * sample_ratio)
    return random.sample(documents, sample_size)


def process_json_files(folder_path: str) -> List[str]:
    print("reading and preprocessing data")
    all_contents = []
    for file_path in json_files_generator(folder_path):
        data = read_json_file(file_path)
        contents = extract_contents(data)
        all_contents.extend(contents)
    return all_contents


def main(input_folder, optimization):

    # Pass the generator directly to Top2Vec
    contents = process_json_files(input_folder)

    # Randomly sample half of the documents
    sampled_contents = random_sample_documents(contents)

    print("Training top2vec")
    model = Top2Vec(documents=sampled_contents, embedding_model="universal-sentence-encoder")

    # Save the trained model
    print("Saving top2vec model")
    model.save("top2vec_model")

    num_topics = model.get_num_topics()
    print("Number of topics NOT reduced", num_topics)

    num_parent_topics = parent_topics(model, num_topics)

    print(num_parent_topics)

    # # Print parent topics
    # parent_topic_words, parent_word_scores, parent_topic_nums = model.get_topics(num_parent_topics, reduced=True)
    # for topic in parent_topic_words:
    #     print(topic)

    # Visualize using UMAP
    umap_vis(model, num_parent_topics, optimization=optimization)

    # Generate topics cloud (the input number is the topic)
    model.generate_topic_wordcloud(0)


if __name__ == "__main__":

    English_json = "./publish/English/Documents/Json"

    main(input_folder=English_json, optimization=True)

