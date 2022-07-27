import uuid

from code.preprocessing import normalizer_interface
from code.utils import io_util


class Cord19Normalizer(normalizer_interface.NormalizerInterface):
    MAP_KEY__CORD19__PAPER_ID = "paper_id"

    def normalize_input_doc(self, input_path, output_path):
        input_article_map = io_util.read_json(input_path)
        normalized_article_map = {}
        # 1- set the document id
        self.set_ids(input_article_map, normalized_article_map)
        # 2- read and set the metadata
        # 3- iterate through tht paragraph text and set the paragraph info - consider special normalization here and cleaning

    def set_ids(self, input_article_map, normalized_article_map):
        # generate a unique id different from the original paper id
        uid = uuid.uuid1()
        normalized_article_map[self.MAP_KEY__UID] = uid
        normalized_article_map[self.MAP_KEY__DOC_ID] = input_article_map[self.MAP_KEY__CORD19__PAPER_ID]
