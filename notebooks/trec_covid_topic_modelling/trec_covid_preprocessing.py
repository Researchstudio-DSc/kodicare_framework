import csv
import sys
from tqdm import tqdm
import hydra
import spacy
from spacy.lang.en import English
from typing import List
import re

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
                    #result.append("<NUM>")
                    continue
                else:
                    result.append(token.lemma_.lower())
    if sent_nlp:
        results = merge_sents_to_doc(results, sents_per_doc)
    return results


no_num_clean_p = re.compile(r'[^\w\s]+|\d+', re.UNICODE)

def preprocess_lm(texts, sent_nlp: spacy.language.Language=None):
    texts, sents_per_doc = get_sentence_split(sent_nlp, texts)
    results = []
    for doc in tqdm(texts, desc="pre-lm"):
        clean_string = no_num_clean_p.sub(' ', doc)
        results.append(clean_string.lower().split())
    results = merge_sents_to_doc(results, sents_per_doc, keep_sentences=True)
    return results


clean_p = re.compile(r'[^\w\s?!.,$%/(){}\[\]:#+\-]+', re.UNICODE)

def preprocess_bert(texts, sent_nlp: spacy.language.Language=None, token_limit=128):
    texts, sents_per_doc = get_sentence_split(sent_nlp, texts)
    results = []
    for doc in tqdm(texts, desc="pre-bert"):
        clean_string = clean_p.sub(' ', doc)
        results.append(clean_string.lower().split())
    results = merge_sents_to_doc(results, sents_per_doc, keep_sentences=False, token_limit=token_limit)
    return results


def merge_sents_to_doc(tok_sentences, sents_per_doc, keep_sentences=False, token_limit=None):
    tok_sentences_merged = []
    i = 0
    for sent_count in sents_per_doc:
        doc_full = []
        if keep_sentences: # just append each sentence as a dok
            for j in range(i, sent_count+i):
                doc_full.append(tok_sentences[j])
        elif token_limit: # merge sentences into token-limited passages
            passage = []
            for j in range(i, sent_count+i):
                if len(passage) == 0:
                    passage.extend(tok_sentences[j])
                elif len(passage) + len(tok_sentences[j]) <= token_limit:
                    passage.extend(tok_sentences[j])
                else:
                    doc_full.append(passage)
                    passage = []
            if len(passage) > 0:
                doc_full.append(passage)
        else: # merge all sentences of a document to a full doc
            for j in range(i, sent_count+i):
                doc_full.extend(tok_sentences[j])
        i += sent_count
        tok_sentences_merged.append(doc_full)
    return tok_sentences_merged


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
    

    if cfg.preprocessing.mode == "bert":
        sent_nlp = English()
        sent_nlp.add_pipe('sentencizer')
        sent_nlp.max_length = 10000000

    with open(cfg.tokenized_path, "w") as fp:
        #for batch in read_trec_covid(batch_size=1024):
        for batch in read_trec_covid(cfg.raw_text_path, 
                                     batch_size=cfg.preprocessing.batch_size, 
                                     only_abstract=cfg.preprocessing.only_abstract):
            batch_cord_uid, batch_text = zip(*batch)
            if cfg.preprocessing.mode == "lm":
                batch_processed = preprocess_lm(batch_text, sent_nlp=sent_nlp)
                for cord_uid, doc in zip(batch_cord_uid, batch_processed):
                    for sent in doc:
                        sent_text = " ".join(sent)
                        fp.write(f'{cord_uid},"{sent_text}"\n')
            elif cfg.preprocessing.mode == "lda":
                batch_processed = preprocess(batch_text, nlp=nlp, sent_nlp=sent_nlp)
                for cord_uid, doc in zip(batch_cord_uid, batch_processed):
                    doc_text = " ".join(doc)
                    fp.write(f'{cord_uid},"{doc_text}"\n')
            elif cfg.preprocessing.mode == "bert":
                batch_processed = preprocess_bert(batch_text, sent_nlp=sent_nlp, token_limit=cfg.preprocessing.token_limit)
                for cord_uid, doc in zip(batch_cord_uid, batch_processed):
                    for passage in doc:
                        passage_text = " ".join(passage)
                        fp.write(f'{cord_uid},"{passage_text}"\n')


if __name__ == '__main__':
    main()
