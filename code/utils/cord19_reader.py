import json
import os
from tqdm import tqdm
from lxml import etree
from code.indexing.index_util import Query
from code.models.doc2vec import preprocess_document
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess

from hydra.utils import instantiate

from code.utils.reader_util import CollectionReader

class CORD19BatchReader(CollectionReader):

    def __init__(self, data_dir=None, collection=None, cord_id_title=None, batch_size: int = 1024) -> None:
        super().__init__(data_dir, collection)
        self.batch_size = batch_size
        # mapping between IDs used in qrel_file and files in index
        with open(os.path.join(data_dir, cord_id_title), "r") as fp:
            cord_id_title_l = json.load(fp)

        self.cord_uid_mapping = {}
        for mapping in cord_id_title_l:
            # paper_id field can contain multiple ids separated by ';'
            paper_ids = [p.strip() for p in mapping['paper_id'].split(';')]
            for paper_id in paper_ids:
                self.cord_uid_mapping[paper_id] = mapping['cord_uid']
    

    def iterate(self):
        batch = []
        # go through all documents and return the documents as a batch of documents
        for fname in tqdm(os.listdir(self.collection_path)):
            with open(os.path.join(self.collection_path, fname), 'r') as fp:
                document = json.load(fp)
            # read might return multiple documents, e.g. when they are split into paragraphs
            for index_document in self.read(document):
                batch.append(index_document)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        # finally yield the last documents
        if len(batch) > 0:
            yield batch


class CORD19Reader(CORD19BatchReader):

    def __init__(self, data_dir=None, collection=None, cord_id_title=None, batch_size: int = 1024) -> None:
        super().__init__(data_dir, collection, cord_id_title, batch_size)


    def read(self, document):
        abstract = " ".join([p["text"] for p in document["paragraphs"] if p["section"]["text"] == "Abstract"])
        cord_uid = self.cord_uid_mapping[document["doc_id"]]
        document_obj = {
            "document_id": cord_uid,
            "uid": document["uid"],
            "doc_id": document["doc_id"],
            "title": document["metadata"]["title"],
            "abstract": abstract
        }
        return [(document_obj["uid"],document_obj)]
    

    def to_string(self, document_obj):
        return f"{document_obj['title']}\n{document_obj['abstract']}"


class CORD19ParagraphReader(CORD19BatchReader):

    def __init__(self, data_dir=None, collection=None, cord_id_title=None, batch_size: int = 1024) -> None:
        super().__init__(data_dir, collection, cord_id_title, batch_size)


    def read(self, document):
        paragraph_objs = []
        for idx, p in enumerate(document["paragraphs"]):
            cord_uid = self.cord_uid_mapping[document["doc_id"]]
            document_obj = {
                "document_id": cord_uid,
                "uid": document["uid"],
                "doc_id": document["doc_id"],
                "paragraph_id": f'{document["uid"]}_{idx}',
                "paragraph_number": idx,
                "title": document["metadata"]["title"],
                "section": p["section"]["text"],
                "paragraph_text": p["text"]
            }
            paragraph_objs.append((document_obj["paragraph_id"], document_obj))
        return paragraph_objs
    

    def to_string(self, document_obj):
        return f"{document_obj['title']}\n{document_obj['section']}\n{document_obj['paragraph_text']}"


MODEL_FOLDER = "./models/doc2vec"

class FAISSReader(CORD19Reader):

    def __init__(self, data_dir=None, collection=None, cord_id_title=None, batch_size: int = 1024) -> None:
        super().__init__(data_dir, collection, cord_id_title, batch_size)
        #self.model = instantiate(model_cfg)
        model_path = os.path.join(MODEL_FOLDER, "paragraphs_basic")
        self.model = Doc2Vec.load(model_path)
    

    def read(self, document):
        batch_data = []
        for uid, document_obj in super().read(document):
            processed_doc = preprocess_document(document_obj, self)
            doc_vector = self.model.infer_vector(processed_doc)
            new_document_object = {
                "uid": document_obj["uid"],
                "document_id": document_obj["document_id"]
            }
            batch_data.append((uid, new_document_object, doc_vector))
        return batch_data


class FAISSParagraphReader(CORD19ParagraphReader):

    def __init__(self, data_dir=None, collection=None, cord_id_title=None, batch_size: int = 1024) -> None:
        super().__init__(data_dir, collection, cord_id_title, batch_size)
        #self.model = instantiate(model_cfg)
        model_path = os.path.join(MODEL_FOLDER, "paragraphs_basic")
        self.model = Doc2Vec.load(model_path)
    

    def read(self, document):
        batch_data = []
        for uid, document_obj in super().read(document):
            processed_doc = preprocess_document(document_obj, self)
            doc_vector = self.model.infer_vector(processed_doc)
            new_document_object = {
                "uid": document_obj["uid"],
                "document_id": document_obj["document_id"]
            }
            batch_data.append((uid, new_document_object, doc_vector))
        return batch_data


class ESQueryReader:

    def __init__(self, queries, data_dir, index_fields) -> None:
        self.queries_path = os.path.join(data_dir, queries)
        self.index_fields = index_fields
    

    def read(self):
        with open(self.queries_path, 'r') as fp:
            topics = etree.parse(fp).getroot()
        queries = []
        for topic in topics:
            query = topic[0]
            queries.append(Query(
                id=topic.attrib['number'],
                data=query.text
            ))
        return queries
    

    def build(self, query, size):
        data = {
            "from" : 0, 
            "size" : size,
            "query": {
                "multi_match" : {
                    "query": query,
                    "type": "cross_fields",
                    "fields": [f for f in self.index_fields]
                }
            }
        }
        return data


class FAISSQueryReader:

    def __init__(self, queries, data_dir) -> None:
        self.queries_path = os.path.join(data_dir, queries)
        #self.model = instantiate(model_cfg)
        model_path = os.path.join(MODEL_FOLDER, "paragraphs_basic")
        self.model = Doc2Vec.load(model_path)
    

    def read(self):
        with open(self.queries_path, 'r') as fp:
            topics = etree.parse(fp).getroot()

        queries = []
        for idx, topic in enumerate(topics):
            query = topic[0]
            question = topic[1]
            assert query.tag == "query"
            assert question.tag == "question"
            #query_vector = self.model.get_vector(f"{query.text} {question.text}")
            processed_doc = simple_preprocess(f"{query.text} {question.text}")
            query_vector = self.model.infer_vector(processed_doc)
            queries.append(Query(
                id=idx,
                data=query_vector
            ))
        return queries
    
