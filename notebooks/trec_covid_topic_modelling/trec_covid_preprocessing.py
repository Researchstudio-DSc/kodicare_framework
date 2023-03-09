import csv
import sys
from tqdm import tqdm
import hydra
import spacy
from spacy.lang.en import English
from typing import List

csv.field_size_limit(sys.maxsize)


def read_trec_covid(f_path, batch_size = None, only_abstract=False):
    with open(f_path, "r") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        keys = reader.__next__()
        batch = []
        for line in reader:
            d = {k:v for k,v in zip(keys, line)}
            cord_uid = d["cord_uid"]
            if only_abstract:
                doc_text = f'{d["title"]} {d["abstract"]}'
            else:
                doc_text = f'{d["title"]} {d["abstract"]} {d["content"]} {d["text"]}'
            doc_text = doc_text.replace("[", " ")
            doc_text = doc_text.replace('"', " ")
            if batch_size:
                batch.append((cord_uid, doc_text))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            else:
                yield cord_uid, doc_text
        
        if len(batch) > 0:
            yield batch


def preprocess(texts, nlp: spacy.language.Language, sent_nlp: spacy.language.Language=None):
    results = []
    if sent_nlp:
        texts, sents_per_doc = get_sentence_split(sent_nlp, texts)
    for doc in tqdm(nlp.pipe(texts), desc="pre"):
        result = []
        results.append(result)
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 3:
                if token.like_num:
                    result.append("<NUM>")
                else:
                    result.append(token.lemma_.lower())
    if sent_nlp:
        results = merge_sents_to_doc(results, sents_per_doc)
    return results


def merge_sents_to_doc(results, sents_per_doc):
    results_merged = []
    i = 0
    for sent_count in sents_per_doc:
        doc_full = []
        for j in range(i, sent_count+i):
            doc_full.extend(results[j])
        i += sent_count
        results_merged.append(doc_full)
    return results_merged


def get_sentence_split(sent_nlp: spacy.language.Language, long_text_batch: List[str]):
    sentences = []
    sents_per_doc = []
    for doc in tqdm(sent_nlp.pipe(long_text_batch), desc="sent"):
        sents = [sent.text.strip() for sent in doc.sents]
        sentences.extend(sents)
        sents_per_doc.append(len(sents))
    return sentences, sents_per_doc





@hydra.main(version_base=None, config_path="./conf", config_name=None)
def main(cfg):
    nlp = spacy.load("en_core_web_md")

    if not cfg.preprocessing.only_abstract:
        sent_nlp = English()
        sent_nlp.add_pipe('sentencizer')
        sent_nlp.max_length = 10000000
    else:
        sent_nlp = None
    with open(cfg.tokenized_path, "w") as fp:
        #for batch in read_trec_covid(batch_size=1024):
        for batch in read_trec_covid(cfg.raw_text_path, 
                                     batch_size=cfg.preprocessing.batch_size, 
                                     only_abstract=cfg.preprocessing.only_abstract):
            batch_cord_uid, batch_text = zip(*batch)
            batch_processed = preprocess(batch_text, nlp=nlp, sent_nlp=sent_nlp)
            for cord_uid, doc in zip(batch_cord_uid, batch_processed):
                doc_text = " ".join(doc)
                fp.write(f'{cord_uid},"{doc_text}"\n')


if __name__ == '__main__':
    main()
