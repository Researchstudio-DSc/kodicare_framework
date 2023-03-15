import csv
import os
import sys
import gensim
from gensim import models
import hydra
csv.field_size_limit(sys.maxsize)


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





def create_corpus(docs, dictionary):
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf


@hydra.main(version_base=None, config_path="./conf", config_name=None)
def main(cfg):
    processed_docs = read_tokenized(cfg.tokenized_path, batch_size=None)

    dictionary = gensim.corpora.Dictionary.load(cfg.dictionary_path)
    
    corpus = create_corpus(processed_docs, dictionary)

    lda_model_tfidf = gensim.models.LdaMulticore(
        corpus, 
        num_topics=cfg.lda.num_topics, 
        id2word=dictionary, 
        passes=cfg.lda.passes, 
        iterations=cfg.lda.iterations,
        minimum_probability=cfg.lda.minimum_probability,
        random_state=cfg.lda.random_state,
        workers=cfg.lda.workers)
    
    lda_model_tfidf.save(cfg.model_path)


if __name__ == '__main__':
    main()