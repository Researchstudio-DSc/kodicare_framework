import csv
import sys
from tqdm import tqdm
csv.field_size_limit(sys.maxsize)


def read_tokenized(path, batch_size = None):
    with open(path, "r") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        batch = []
        #for line in tqdm(reader, desc="batch"):
        for line in tqdm(reader):
            cord_uid, doc_text_tokenized = line
            doc_tokens = doc_text_tokenized.split(" ")
            # remove num token after all
            doc_tokens = [t for t in doc_tokens if t != "<NUM>"]
            if batch_size:
                batch.append(doc_tokens)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            else:
                yield doc_tokens
        
        if len(batch) > 0:
            yield batch