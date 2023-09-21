from quant_pre_utils import *

from top2vec import Top2Vec
import re
import re
import nltk
from nltk.corpus import stopwords


def main(input_folder, optimization):

    list_json = list_files_in_dir(input_folder)
    first = list_json[0] # 1 collection of documents
    documents = read_json(join(input_folder, first))

    print(documents)

    documents = documents.contents.tolist()

    # Preprocess and clean the documents
    documents_cleaned = [preprocess_and_clean_text(doc) for doc in documents]

    # Initialize a Top2Vec model
    model = Top2Vec(documents_cleaned, embedding_model='universal-sentence-encoder')

    # Save the trained model
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

