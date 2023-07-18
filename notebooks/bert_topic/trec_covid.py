from bertopic import BERTopic
import glob
import time
import random
import re
from tqdm import tqdm
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import numpy as np
import pandas as pd
import csv
random_state = 42
random.seed(random_state)

f_tokenized_path = "/home/tfink/data/kodicare/trec-covid/dtc_evolving_bert/0.txt"
f_tokenized_other_path = "/home/tfink/data/kodicare/trec-covid/dtc_evolving_bert/11.txt"


def read_cleaned(path):
    with open(path, "r") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        passages = []
        passage_uids = []
        for line in tqdm(reader):
            cord_uid, passage_text_cleaned = line
            passages.append(passage_text_cleaned)
            passage_uids.append(cord_uid)
        return passages, passage_uids


passages, passage_uids = read_cleaned(f_tokenized_path)


min_topic_size = 5
min_samples = None
cluster_selection_epsilon = 0.25
# dimensionality reduction
umap_model = umap.UMAP(n_neighbors=15, 
                       n_components=5, 
                       min_dist=0.0, 
                       metric='cosine', 
                       random_state=random_state, 
                       verbose=True)
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_topic_size,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        prediction_data=True,
                        cluster_selection_epsilon=cluster_selection_epsilon,
                        min_samples=min_samples)
#hdbscan_model=None

# vectorizer
vectorizer = CountVectorizer(stop_words='english')

# create BerTopic model
#embedding_model = "xlm-r-bert-base-nli-stsb-mean-tokens"
#embedding_model = "all-mpnet-base-v2"
embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
topic_model = BERTopic(embedding_model=embedding_model, 
                       umap_model=umap_model, 
                       hdbscan_model=hdbscan_model,
                       vectorizer_model=vectorizer,
                       language="english", 
                       calculate_probabilities=False, 
                       min_topic_size=min_topic_size,
                       verbose=True)

# Perform topic modeling with BERTopic
topics, probabilities = topic_model.fit_transform(passages[:100000])


topic_info = topic_model.get_topic_info()

topic2desc = {}
for index, row in topic_info.iterrows():
    if row.Topic == -1:
        continue
    topic2desc[row.Topic] = row.Name


#passages, passage_uids = read_cleaned(f_tokenized_other_path)
other_topics, _ = topic_model.transform(passages[100000:200000])


def get_topic_proportions(topics):
    # first calculate base topic counts and proportions
    outlier_count = 0
    topic_counts = {topic:0 for topic in range(len(topic2desc))}
    topic_proportions = {}
    non_outlier_docs = 0
    for idx, topic in enumerate(topics):
        if topic == -1:
            outlier_count += 1
            continue
        non_outlier_docs += 1
        topic_counts[topic] += 1
    for topic in range(len(topic2desc)):
        topic_proportions[topic] = topic_counts[topic] / non_outlier_docs
    outlier_proportion = outlier_count / len(topics)
    return topic_proportions, outlier_proportion


def get_intersection(topic_proportions_a, topic_proportions_b):
    total = 0
    for topic_desc in topic_proportions_a.keys():
        total += min(topic_proportions_a[topic_desc], topic_proportions_b[topic_desc])
    return total


def calculate_topic_intersection(base_topics, other_topics):
    base_topic_proportions, base_outlier_proportion = get_topic_proportions(base_topics)
    other_topic_proportions, other_outlier_proportion = get_topic_proportions(other_topics)

    # calc
    intersection = get_intersection(base_topic_proportions, other_topic_proportions)
    print(f"base_outliers: {base_outlier_proportion:.2%}, other_outliers: {other_outlier_proportion:.2%}, intersection: {intersection:.2%}")


base_topic_proportions, base_outlier_proportion = get_topic_proportions(topics)
other_topic_proportions, other_outlier_proportion = get_topic_proportions(other_topics)

# calc
intersection = get_intersection(base_topic_proportions, other_topic_proportions)
print(f"base_outliers: {base_outlier_proportion:.2%}, other_outliers: {other_outlier_proportion:.2%}, intersection: {intersection:.2%}")

df = pd.DataFrame({
    "Topics": ["outliers"] + [str(topic) for topic in range(len(topic2desc))],
    "Topics_Names": ["outliers"] + [topic2desc[topic] for topic in range(len(topic2desc))],
    'Base': [base_outlier_proportion]+[base_topic_proportions[topic] for topic in range(len(topic2desc))],
    'Other': [other_outlier_proportion]+[other_topic_proportions[topic] for topic in range(len(topic2desc))],
})

fig = px.bar(
    data_frame = df,
    x = "Topics",
    y = ["Base","Other"],
    opacity = 0.9,
    orientation = "v",
    barmode = 'group',
    title='Topic Proportions',
    hover_data=["Topics_Names"],
    color_discrete_sequence=px.colors.qualitative.D3
    #color_discrete_sequence=px.colors.sequential.Inferno_r
)

fig.write_html("trec_covid.html")