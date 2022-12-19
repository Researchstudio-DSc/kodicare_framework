
import os
from typing import Iterable, Union


class CollectionReader:
    # Iterate through a collection and yield indexable index_documents
    # or batches of index_documents
    # An index_document should be indexed as-is
    # All necessary pre-index transformations should be done by the Reader

    def __init__(self, data_dir=None, collection=None) -> None:
        # collection path can be either a dir or a file
        self.data_dir = data_dir
        if collection:
            self.collection_path = os.path.join(data_dir, collection)


    def read(self, document) -> Union[Iterable, tuple]:
        # returns either a document tuple (doc_id, doc_data)
        # or a list of such tuples 
        # (in case a "document" consists of smaller units e.g. passages)
        raise NotImplementedError
    

    def iterate(self) -> Union[Iterable, object]:
        # should either iterate through batches
        # or single index_documents
        raise NotImplementedError