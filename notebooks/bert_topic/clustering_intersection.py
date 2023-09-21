import random
from tqdm import tqdm
import umap
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import numpy as np
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import BisectingKMeans
random_state = 42
#random_state = 420
random.seed(random_state)

f_tokenized_path = "/home/tfink/data/kodicare/trec-covid/dtc_evolving_bert/0.csv"
f_tokenized_other_path = "/home/tfink/data/kodicare/trec-covid/dtc_evolving_bert/11.csv"

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


def get_topic_proportions(n_clusters, topics):
    # first calculate base topic counts and proportions
    outlier_count = 0
    topic_counts = {topic:0 for topic in range(n_clusters)}
    topic_proportions = {}
    non_outlier_docs = 0
    for idx, topic in enumerate(topics):
        if topic == -1:
            outlier_count += 1
            continue
        non_outlier_docs += 1
        topic_counts[topic] += 1
    for topic in range(n_clusters):
        topic_proportions[topic] = topic_counts[topic] / non_outlier_docs
    outlier_proportion = outlier_count / len(topics)
    return topic_proportions, outlier_proportion


def get_intersection(n_clusters, topic_proportions_a, topic_proportions_b):
    total = 0
    for topic in range(n_clusters):
        total += min(topic_proportions_a[topic], topic_proportions_b[topic])
    return total



def main():
    passages, passage_uids = read_cleaned(f_tokenized_path)
    embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(embedding_model)
    passages_encoded = model.encode(passages, show_progress_bar=True, batch_size=32)

    # dimensionality reduction
    umap_model = umap.UMAP(n_neighbors=15, 
                        n_components=5, 
                        min_dist=0.0, 
                        metric='cosine', 
                        random_state=random_state, 
                        verbose=True)
    embedding = umap_model.fit_transform(passages_encoded)

    n_clusters = int(np.sqrt(len(embedding)))
    clustering_model = BisectingKMeans(n_clusters=n_clusters, 
                        n_init=1,
                        bisecting_strategy='largest_cluster',
                        random_state=random_state)
    topics = clustering_model.fit_predict(embedding)

    passages, passage_uids = read_cleaned(f_tokenized_other_path)
    passages_encoded = model.encode(passages, show_progress_bar=True, batch_size=32)
    embedding = umap_model.transform(passages_encoded)
    other_topics = clustering_model.predict(embedding)

    base_topic_proportions, base_outlier_proportion = get_topic_proportions(n_clusters, topics)
    other_topic_proportions, other_outlier_proportion = get_topic_proportions(n_clusters, other_topics)

    # calc
    intersection = get_intersection(n_clusters, base_topic_proportions, other_topic_proportions)
    print(f"base_outliers: {base_outlier_proportion:.2%}, other_outliers: {other_outlier_proportion:.2%}, intersection: {intersection:.2%}")

    df = pd.DataFrame({
        "Topics": ["outliers"] + [str(topic) for topic in range(n_clusters)],
        'Base': [base_outlier_proportion]+[base_topic_proportions[topic] for topic in range(n_clusters)],
        'Other': [other_outlier_proportion]+[other_topic_proportions[topic] for topic in range(n_clusters)],
    })

    fig = px.bar(
        data_frame = df,
        x = "Topics",
        y = ["Base","Other"],
        opacity = 0.9,
        orientation = "v",
        barmode = 'group',
        title='Topic Proportions',
        color_discrete_sequence=px.colors.qualitative.D3
        #color_discrete_sequence=px.colors.sequential.Inferno_r
    )
    fig.update_yaxes(range=[0.0, 0.035])
    fig.write_html("trec_covid.html")


if __name__ == "__main__":
    main()