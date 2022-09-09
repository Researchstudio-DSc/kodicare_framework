import os
import argparse
from code.indexing.readers import CORD19Reader, CORD19ParagraphReader

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

MODEL_FOLDER = "./models/doc2vec"


def preprocess_document(raw_document, reader):
    doc_s = reader.to_string(raw_document)
    return simple_preprocess(doc_s)



def stream_documents(batches, reader):
    i = 0
    for batch in batches:
        for doc_uid, doc in batch:
            processed_doc = preprocess_document(doc, reader)
            yield TaggedDocument(processed_doc, [i])
            i += 1


class DocumentStream:

    def __init__(self, data_folder, reader):
        self.data_folder = data_folder
        self.reader = reader
    

    def __iter__(self):
        batches = self.reader.iterate(self.data_folder)
        self.stream = stream_documents(batches, self.reader)
        return self
    
    
    def __next__(self):
        return self.stream.__next__()
        


def main(args):

    if args.index_type == "paragraphs":
        reader = CORD19ParagraphReader(batch_size=16384)
    else:
        reader = CORD19Reader(batch_size=1024)

    stream = DocumentStream(args.data_folder, reader)

    #model = Doc2Vec(documents=stream, vector_size=300, window=5, min_count=3, workers=4)
    ## Dummy input for testing
    common_texts = [["system", "response"], ["system", "irresponsible"]]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    model = Doc2Vec(documents=documents, vector_size=300, window=5, min_count=1, workers=4)

    model_path = os.path.join(MODEL_FOLDER, args.model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}")
    vector = model.infer_vector(["test", "edit"])
    print(vector)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for training a Doc2Vec model for retrieval'
    )
    parser.add_argument('--index_type', help='paragraphs or doc')
    parser.add_argument('--data_folder', help='Folder containing the files that should be indexed')
    parser.add_argument('--model_name', help='Name of the Doc2Vec model')

    args = parser.parse_args()
    main(args)