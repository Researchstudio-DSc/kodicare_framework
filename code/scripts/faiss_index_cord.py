import argparse
import os
from code.indexing.faiss_index import Index
from code.indexing.readers import CORD19Reader, CORD19ParagraphReader
from code.models.doc2vec import preprocess_document
from gensim.models.doc2vec import Doc2Vec

MODEL_FOLDER = "./models/doc2vec"

# CORD19 specific iterator
def iterate(model, batches, reader):
    for batch in batches:
        batch_data = []
        for uid, document_obj in batch:
            processed_doc = preprocess_document(document_obj, reader)
            doc_id = document_obj['doc_id'] # CORD19 doc_id
            doc_vector = model.infer_vector(processed_doc)
            batch_data.append((uid, doc_id, doc_vector))
        yield batch_data


def main(args):
    if args.index_type == "paragraphs":
        reader = CORD19ParagraphReader(batch_size=16384)
    else:
        reader = CORD19Reader(batch_size=1024)
    index = Index(args.index_name)

    index.create_index(vector_size=args.vector_size)

    model_path = os.path.join(MODEL_FOLDER, args.model_name)
    model = Doc2Vec.load(model_path)

    batches = reader.iterate(args.data_folder)

    index.index_docs(iterate(model, batches, reader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for testing the elasticsearch index in es_index'
    )
    parser.add_argument('--index_name', help='The name of the index')
    parser.add_argument('--index_type', help='paragraphs or doc')
    parser.add_argument('--data_folder', help='Folder containing the files that should be indexed')
    parser.add_argument('--model_name', help='Name of the Doc2Vec model')
    parser.add_argument('--vector_size', type=int, help='Dimensions of the Doc2Vec model')

    args = parser.parse_args()
    main(args)