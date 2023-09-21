import sys
from tqdm import tqdm
import hydra
import spacy
from spacy.lang.en import English
from typing import List
import re
from pathlib import Path

import json


def read_robust(f_path, batch_size = None):
    with open(f_path, "r") as fp:
        data = json.load(fp)
        batch = []
        for doc_data in data:
            doc_id = doc_data["id"]
            doc_text = f'{doc_data["title"]} {doc_data["contents"]}'
            doc_text = doc_text.replace("[", " ")
            doc_text = doc_text.replace('"', " ")
            if batch_size:
                batch.append((doc_id, doc_text))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            else:
                yield doc_id, doc_text
        
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
                    #result.append("<NUM>")
                    continue
                else:
                    result.append(token.lemma_.lower())
    if sent_nlp:
        results = merge_sents_to_doc(results, sents_per_doc)
    return results


clean_p = re.compile(r'[^\w\s]+|\d+', re.UNICODE)

def preprocess_lm(texts, sent_nlp: spacy.language.Language=None):
    texts, sents_per_doc = get_sentence_split(sent_nlp, texts)
    results = []
    for doc in tqdm(texts, desc="pre-lm"):
        clean_string = clean_p.sub(' ', doc)
        results.append(clean_string.lower().split())
    results = merge_sents_to_doc(results, sents_per_doc, keep_sentences=True)
    return results


def merge_sents_to_doc(results, sents_per_doc, keep_sentences=False):
    results_merged = []
    i = 0
    for sent_count in sents_per_doc:
        doc_full = []
        if keep_sentences:
            for j in range(i, sent_count+i):
                doc_full.append(results[j])
        else:
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


def preprocess_file(cfg, in_path, out_path, sent_nlp=None, nlp=None):
    print(str(in_path))
    with open(out_path, "w") as fp:
        for batch in read_robust(in_path, 
                                batch_size=cfg.preprocessing.batch_size):
            batch_doc_id, batch_text = zip(*batch)
            if cfg.preprocessing.mode == "lm":
                batch_processed = preprocess_lm(batch_text, sent_nlp=sent_nlp)
                for _, doc in zip(batch_doc_id, batch_processed):
                    for sent in doc:
                        sent_text = " ".join(sent)
                        fp.write(f'{sent_text}\n')
            elif cfg.preprocessing.mode == "lda":
                batch_processed = preprocess(batch_text, nlp=nlp, sent_nlp=sent_nlp)
                for doc_id, doc in zip(batch_doc_id, batch_processed):
                    doc_text = " ".join(doc)
                    fp.write(f'{doc_id},"{doc_text}"\n')


@hydra.main(version_base=None, config_path="./conf", config_name=None)
def main(cfg):

    if cfg.preprocessing.mode == "lda":
        nlp = spacy.load("en_core_web_md")

        if not cfg.preprocessing.only_abstract:
            sent_nlp = English()
            sent_nlp.add_pipe('sentencizer')
            sent_nlp.max_length = 10000000
        else:
            sent_nlp = None
    if cfg.preprocessing.mode == "lm":
        sent_nlp = English()
        sent_nlp.add_pipe('sentencizer')
        sent_nlp.max_length = 10000000

    corpora_path = Path(cfg.corpora)
    json_path = Path(cfg.json)
    if json_path.is_dir():
        for f in json_path.glob("*.json"):
            f_out = (corpora_path / f.name).with_suffix(".txt")
            preprocess_file(cfg, in_path=f, out_path=f_out, sent_nlp=sent_nlp)


if __name__ == '__main__':
    main()
