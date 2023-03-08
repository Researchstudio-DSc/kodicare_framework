import csv
import os
import sys
import gensim
from gensim import models
import numpy as np
csv.field_size_limit(sys.maxsize)

f_tokenized_path = "/home/tfink/data/kodicare/trec-covid/dtc_evolving/0.csv"
num_topics = 30
passes = 40
iterations = 5000
minimum_probability = 0.05


def read_tokenized(path, batch_size = None):
    with open(path, "r") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        batch = []
        #for line in tqdm(reader, desc="batch"):
        for line in reader:
            cord_uid, doc_text_tokenized = line
            doc_tokens = doc_text_tokenized.split(" ")
            if batch_size:
                batch.append(doc_tokens)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            else:
                yield doc_tokens
        
        if len(batch) > 0:
            yield batch


processed_docs = list(read_tokenized(f_tokenized_path, batch_size=None))
dictionary = gensim.corpora.Dictionary(processed_docs+processed_docs_comp)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


def create_corpus(docs):
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf


corpus = create_corpus(processed_docs)

lda_model_tfidf = gensim.models.LdaMulticore(
    corpus, 
    num_topics=num_topics, 
    id2word=dictionary, 
    passes=passes, 
    iterations=iterations,
    minimum_probability=minimum_probability,
    random_state=2018,
    workers=4)


for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


lda_1_path = os.path.join(model_dir, "lda_1")
lda_model_tfidf.save(lda_1_path)