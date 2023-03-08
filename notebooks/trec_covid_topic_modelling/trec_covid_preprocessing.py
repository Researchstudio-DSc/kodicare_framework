import csv
import sys
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

f_path = "./data/TREC-COVID_complete_content.csv"
f_out_path = "./data/TREC-COVID_complete_content.csv.tokenized.txt"


def read_trec_covid(batch_size = None):
    with open(f_path, "r") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        keys = reader.__next__()
        batch = []
        #for line in tqdm(reader, desc="batch"):
        for line in reader:
            d = {k:v for k,v in zip(keys, line)}
            cord_uid = d["cord_uid"]
            #print(cord_uid)
            doc_text = f'{d["title"]} {d["abstract"]} {d["content"]} {d["text"]}'
            doc_text = doc_text.replace("[", " ")
            doc_text = doc_text.replace('"', " ")
            if batch_size:
                batch.append((cord_uid, doc_text))
                #batch.append((cord_uid, "doc_text"))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            else:
                yield cord_uid, doc_text
        
        if len(batch) > 0:
            yield batch



import spacy
from spacy.lang.en import English
nlp = spacy.load("en_core_web_md")
#nlp.max_length = 10000000

sent_nlp = English()
sent_nlp.add_pipe('sentencizer')
sent_nlp.max_length = 10000000


def preprocess(texts, sentence_split=False):
    results = []
    if sentence_split:
        texts, sents_per_doc = get_sentence_split(texts)
    for doc in tqdm(nlp.pipe(texts), desc="pre"):
        result = []
        results.append(result)
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 3:
                if token.like_num:
                    result.append("<NUM>")
                else:
                    result.append(token.lemma_.lower())
    if sentence_split:
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


def get_sentence_split(long_text_batch):
    sentences = []
    sents_per_doc = []
    for doc in tqdm(sent_nlp.pipe(long_text_batch), desc="sent"):
        sents = [sent.text.strip() for sent in doc.sents]
        sentences.extend(sents)
        sents_per_doc.append(len(sents))
    return sentences, sents_per_doc



with open(f_out_path, "w") as fp:
    #for batch in read_trec_covid(batch_size=16384):
    for batch in read_trec_covid(batch_size=1024):
        batch_cord_uid, batch_text = zip(*batch)
        batch_processed = preprocess(batch_text, sentence_split=True)
        for cord_uid, doc in zip(batch_cord_uid, batch_processed):
            doc_text = " ".join(doc)
            fp.write(f'{cord_uid},"{doc_text}"\n')
