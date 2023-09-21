import csv
import os
import sys
import gensim
from gensim import models
import matplotlib.pyplot as plt
import numpy as np
from notebooks.trec_covid_topic_modelling.util import read_tokenized
from tqdm import tqdm
import plotly.express as px
csv.field_size_limit(sys.maxsize)

model_dir = "models/trec_covid_topic_modelling"

dictionary_path = os.path.join(model_dir, "dict1")

#model_1_path = os.path.join(model_dir, "lda1_comp")
model_1_path = os.path.join(model_dir, "lda1")
#model_2_path = os.path.join(model_dir, "lda_comp")
model_2_path = os.path.join(model_dir, "lda2")

model_1 = gensim.models.LdaMulticore.load(model_1_path)
model_2 = gensim.models.LdaMulticore.load(model_2_path)

corpus_1_path = "/home/tfink/data/kodicare/trec-covid/dtc_evolving/0.csv"
corpus_2_path = "/home/tfink/data/kodicare/trec-covid/dtc_evolving/11.csv"
#corpus_2_path = "data/trec_covid_topic_modelling/abcnews-date-text.csv.tokenized.txt"


def plot_difference_matplotlib(mdiff, title="", annotation=None):
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='inferno_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)
    plt.show()


def create_corpus(docs, dictionary):
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf


def get_corpus_document_dist(model, corpus):
    topic_probs_sum = np.zeros((model.num_topics,), dtype=np.float32)

    topics_corpus = model.get_document_topics(corpus)

    for topics in tqdm(topics_corpus):
        for topic_id, prob in topics:
            topic_probs_sum[topic_id] += prob

    topic_probs = topic_probs_sum/len(topics_corpus)
    return topic_probs

dictionary = gensim.corpora.Dictionary.load(dictionary_path)


processed_docs = read_tokenized(corpus_1_path, batch_size=None)
corpus = create_corpus(processed_docs, dictionary)


lda1_c1_topics = get_corpus_document_dist(model_1, corpus)
lda2_c1_topics = get_corpus_document_dist(model_2, corpus)

processed_docs = read_tokenized(corpus_2_path, batch_size=None)
corpus = create_corpus(processed_docs, dictionary)

lda1_c2_topics = get_corpus_document_dist(model_1, corpus)
lda2_c2_topics = get_corpus_document_dist(model_2, corpus)


fig = px.bar({"native_corpus":lda1_c1_topics, "other_corpus":lda1_c2_topics}, barmode='group')
fig.write_html("lda1.html")

fig = px.bar({"native_corpus":lda2_c2_topics, "other_corpus":lda2_c1_topics}, barmode='group')
fig.write_html("lda2.html")