import csv
import os
import sys
import gensim
from gensim import models
import hydra
from util import read_tokenized
csv.field_size_limit(sys.maxsize)


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