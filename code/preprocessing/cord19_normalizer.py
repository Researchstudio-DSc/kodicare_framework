import uuid

from code.preprocessing import normalizer_interface
from code.utils import io_util


class Cord19Normalizer(normalizer_interface.NormalizerInterface):
    MAP_KEY__CORD19__PAPER_ID = "paper_id"
    MAP_KEY__CORD19__METADATA = "metadata"
    MAP_KEY__CORD19__TITLE = "title"
    MAP_KEY__CORD19__AUTHORS = "authors"
    MAP_KEY__CORD19__FIRST_NAME = "first"
    MAP_KEY__CORD19__MIDDLE_NAME = "middle"
    MAP_KEY__CORD19__LAST_NAME = "last"
    MAP_KEY__CORD19__AFFILIATION = "affiliation"
    MAP_KEY__CORD19__LOCATION = "location"
    MAP_KEY__CORD19__SETTLEMENT = "settlement"
    MAP_KEY__CORD19__REGION = "region"
    MAP_KEY__CORD19__INSTITUTION = "institution"
    MAP_KEY__CORD19__EMAIL = "email"

    def normalize_input_doc(self, input_path, output_path):
        input_article_map = io_util.read_json(input_path)
        normalized_article_map = {}
        # 1- set the document id
        self.set_ids(input_article_map, normalized_article_map)
        # 2- read and set the metadata
        self.set_metadata(input_article_map, normalized_article_map)
        # 3- iterate through tht paragraph text and set the paragraph info - consider special normalization here and cleaning

        # write to json
        io_util.write_json(output_path, normalized_article_map)

    def set_ids(self, input_article_map, normalized_article_map):
        # generate a unique id different from the original paper id
        uid = str(uuid.uuid1())
        normalized_article_map[self.MAP_KEY__UID] = uid
        normalized_article_map[self.MAP_KEY__DOC_ID] = input_article_map[self.MAP_KEY__CORD19__PAPER_ID]

    def set_metadata(self, input_article_map, normalized_article_map):
        normalized_article_map[self.MAP_KEY__METADATA] = {}
        normalized_article_map[self.MAP_KEY__METADATA][self.MAP_KEY__TITLE] \
            = input_article_map[self.MAP_KEY__CORD19__METADATA][self.MAP_KEY__CORD19__TITLE]
        normalized_article_map[self.MAP_KEY__METADATA][self.MAP_KEY__DOC_TYPE] = self.DOC_TYPE_SCIENTIFIC_PAPER
        normalized_article_map[self.MAP_KEY__METADATA][self.MAP_KEY__AUTHORS] \
            = self.construct_authors_list(input_article_map, normalized_article_map)
        normalized_article_map[self.MAP_KEY__METADATA][self.MAP_KEY__PUBLISHER] = ""
        normalized_article_map[self.MAP_KEY__METADATA][self.MAP_KEY__YEAR] = ""

    def construct_authors_list(self, input_article_map, normalized_article_map):
        normalized_authors = []
        for author in input_article_map[self.MAP_KEY__CORD19__METADATA][self.MAP_KEY__CORD19__AUTHORS]:
            normalized_author = {
                self.MAP_KEY__FIRST_NAME: author[self.MAP_KEY__CORD19__FIRST_NAME],
                self.MAP_KEY__MIDDLE_NAME: ' '.join(author[self.MAP_KEY__CORD19__MIDDLE_NAME]),
                self.MAP_KEY__LAST_NAME: author[self.MAP_KEY__CORD19__LAST_NAME],
                self.MAP_KEY__AFFILIATION: {
                    self.MAP_KEY__INSTITUTION: author[self.MAP_KEY__CORD19__AFFILIATION][
                        self.MAP_KEY__CORD19__INSTITUTION],
                    self.MAP_KEY__ADDRESS: ' '.join(str(x) for x in author[self.MAP_KEY__CORD19__AFFILIATION][
                        self.MAP_KEY__CORD19__LOCATION].values())},
                self.MAP_KEY__EMAIL: author[self.MAP_KEY__CORD19__EMAIL],
            }
            normalized_authors.append(normalized_author)
        return normalized_authors
