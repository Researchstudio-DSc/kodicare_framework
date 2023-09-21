from tqdm import tqdm
import hydra
from pathlib import Path
import subprocess
import time
import multiprocessing
import random
from tqdm import tqdm
import umap
import numpy as np
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import BisectingKMeans
import pickle

def score_corpus(model_path, corpus_path, embedding_model, position):
    # compute the score for the corpus
    with open(model_path, "rb") as fp:
        umap_model, clustering_model, topics, n_clusters = pickle.load(fp)

    passages, passage_uids = read_cleaned(corpus_path)
    model = SentenceTransformer(embedding_model)
    passages_encoded = model.encode(passages, show_progress_bar=True, batch_size=32)
    embedding = umap_model.transform(passages_encoded)
    other_topics = clustering_model.predict(embedding)

    base_topic_proportions, base_outlier_proportion = get_topic_proportions(n_clusters, topics)
    other_topic_proportions, other_outlier_proportion = get_topic_proportions(n_clusters, other_topics)

    # calc
    intersection = get_intersection(n_clusters, base_topic_proportions, other_topic_proportions)
    return corpus_path, intersection


def get_scores(model_path, corpora, embedding_model, processes=4):
    if processes > 1:
        with multiprocessing.Pool(processes=processes) as pool:
            args = [(model_path, comp_corpus, embedding_model, idx) for idx, comp_corpus in enumerate(corpora)]
            scores = pool.starmap(score_corpus, args)
    else:
        scores = [score_corpus(model_path, comp_corpus, embedding_model, idx) for idx, comp_corpus in enumerate(corpora)]

    return scores


def calculate_deltas(corpus, scores, out_fp):
    for comp_corpus, score in scores:
        out_fp.write(f'"{Path(corpus).stem}", "{Path(comp_corpus).stem}", {score:.4f}\n')


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


def create_model(cfg, corpus, model_path):
    passages, passage_uids = read_cleaned(corpus)
    #embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(cfg.embedding_model)
    passages_encoded = model.encode(passages, show_progress_bar=True, batch_size=32)

    # dimensionality reduction
    umap_model = umap.UMAP(n_neighbors=15, 
                        n_components=5, 
                        min_dist=0.0, 
                        metric='cosine', 
                        random_state=cfg.random_state, 
                        verbose=True)
    embedding = umap_model.fit_transform(passages_encoded)

    n_clusters = int(np.sqrt(len(embedding)))
    clustering_model = BisectingKMeans(n_clusters=n_clusters, 
                        n_init=1,
                        bisecting_strategy='largest_cluster',
                        random_state=cfg.random_state)
    topics = clustering_model.fit_predict(embedding)
    with open(model_path, "wb") as fp:
        pickle.dump((umap_model, clustering_model, topics, n_clusters), fp)



@hydra.main(version_base=None, config_path="./conf", config_name=None)
def main(cfg):
    random.seed(cfg.random_state)
    t00 = time.time()

    if "corpora_list" in cfg and cfg.corpora_list:
        corpora = cfg.corpora_list
    elif "corpora" in cfg and cfg.corpora:
        corpora_path = Path(cfg.corpora)
        corpora = list(corpora_path.glob("*.csv"))
    else:
        assert False

    for corpus in corpora:
        print(f'### CORPUS: {corpus}')
        model_path = Path(corpus).with_suffix('.pkl')
        t0 = time.time()
        # create clustering / topic model
        if not model_path.is_file():
            create_model(cfg, corpus, model_path)

        t1 = time.time()
        print('#'*80)
        print(f"Model creation: {t1-t0}")
        # compare corpus to all other corpora and calculate intersection / deltas
        scores = get_scores(str(model_path), corpora, cfg.embedding_model, processes=cfg.processes)
        with open(cfg.results_file, "a") as fp:
            calculate_deltas(corpus, scores, out_fp=fp)
        
        t2 = time.time()
        print('#'*80)
        print(f"KD scoring: {t2-t1}")
        print('#'*80)
        print(f"Iteration time: {t2-t0}")
    print('#'*80)
    print('#'*80)
    print(f"Total time: {t2-t00}")



if __name__ == '__main__':
    main()
