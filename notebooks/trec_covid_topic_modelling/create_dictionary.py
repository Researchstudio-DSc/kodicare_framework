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


@hydra.main(version_base=None, config_path="./conf", config_name=None)
def main(cfg):
    dictionary = gensim.corpora.Dictionary()
    for path in cfg.tokenized_paths:
        dictionary.add_documents(read_tokenized(path, batch_size=None))
    dictionary.filter_extremes(no_below=cfg.dictionary.no_below, 
                               no_above=cfg.dictionary.no_above, 
                               keep_n=cfg.dictionary.keep_n)
    
    dictionary.save(cfg.dictionary_path)


if __name__ == '__main__':
    main()