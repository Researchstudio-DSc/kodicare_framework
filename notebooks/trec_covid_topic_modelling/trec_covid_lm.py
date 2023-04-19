data_dir = "../../data/trec_covid_topic_modelling"
model_dir = "../../models/trec_covid_topic_modelling"

from nltk.lm import MLE, AbsoluteDiscountingInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Vocabulary
import os
import pickle as pkl
from util import read_tokenized

f_tokenized_path = "/home/tfink/data/kodicare/trec-covid/dtc_evolving/0.csv"
f_tokenized_other_path = "/home/tfink/data/kodicare/trec-covid/dtc_evolving/11.csv"

lm_model_path = os.path.join(model_dir, "lm1")

# TREC Covid docs
processed_docs = list(read_tokenized(f_tokenized_path, batch_size=None))
print(processed_docs[0][:10])

train_data, padded_sents = padded_everygram_pipeline(3, processed_docs)
model = AbsoluteDiscountingInterpolated(order=3, vocabulary = Vocabulary(unk_cutoff=5))
print(model.vocab)
model.fit(train_data, padded_sents)
print(model.vocab)


def doc_ngram_iter(doc_ngrams):
        for ngram in doc_ngrams:
            yield ngram


train_data, padded_sents = padded_everygram_pipeline(3, processed_docs)
doc_perplexity = []
for doc in train_data:
    perplexity = model.perplexity(doc_ngram_iter(doc))
    doc_perplexity.append(perplexity)



import numpy as np
p1 = np.mean(doc_perplexity)


processed_docs = list(read_tokenized(f_tokenized_other_path, batch_size=None))

train_data, padded_sents = padded_everygram_pipeline(3, processed_docs)
doc_perplexity = []
for doc in train_data:
    perplexity = model.perplexity(doc_ngram_iter(doc))
    doc_perplexity.append(perplexity)


p2 = np.mean(doc_perplexity)

