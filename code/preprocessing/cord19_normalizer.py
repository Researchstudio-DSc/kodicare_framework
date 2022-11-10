from code.preprocessing import normalizer_interface
from code.utils import io_util


class Cord19Normalizer(normalizer_interface.NormalizerInterface):
    MAP_KEY__CORD19__PAPER_ID = "paper_id"
    MAP_KEY__CORD19__CORD_UID = "cord_uid"
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
    MAP_KEY__CORD19__ABSTRACT = "abstract"
    MAP_KEY__CORD19__SECTION = "section"
    MAP_KEY__CORD19__TEXT = "text"
    MAP_KEY__CORD19__BODY_TEXT = "body_text"
    MAP_KEY__CORD19__CITE_SPANS = "cite_spans"
    MAP_KEY__CORD19__REF_SPANS = "ref_spans"
    MAP_KEY__CORD19__START = "start"
    MAP_KEY__CORD19__END = "end"
    MAP_KEY__CORD19__REF_ID = "ref_id"

    def normalize_input_doc(self, input_path, output_path, metadata=None):
        input_article_map = io_util.read_json(input_path)
        normalized_article_map = {}
        # 1- set the document id
        self.set_ids(input_article_map, normalized_article_map, metadata)
        # 2- read and set the metadata
        self.set_metadata(input_article_map, normalized_article_map)
        # 3- iterate through tht paragraph text and set the paragraph info - consider special normalization here and cleaning
        self.set_paragraphs_list(input_article_map, normalized_article_map)
        # write to json
        io_util.write_json(output_path, normalized_article_map)

    def set_ids(self, input_article_map, normalized_article_map, metadata):
        # use the paper id as the doc and the unique id
        paper_id = input_article_map[self.MAP_KEY__CORD19__PAPER_ID]
        normalized_article_map[self.MAP_KEY__UID] = paper_id
        normalized_article_map[self.MAP_KEY__DOC_ID] = paper_id
        normalized_article_map[self.MAP_KEY__CORD19__CORD_UID] = metadata[paper_id] if paper_id in metadata else ""

    def set_metadata(self, input_article_map, normalized_article_map):
        normalized_article_map[self.MAP_KEY__METADATA] = {}
        normalized_article_map[self.MAP_KEY__METADATA][self.MAP_KEY__TITLE] \
            = input_article_map[self.MAP_KEY__CORD19__METADATA][self.MAP_KEY__CORD19__TITLE]
        normalized_article_map[self.MAP_KEY__METADATA][self.MAP_KEY__DOC_TYPE] = self.DOC_TYPE_SCIENTIFIC_PAPER
        normalized_article_map[self.MAP_KEY__METADATA][self.MAP_KEY__AUTHORS] \
            = self.construct_authors_list(input_article_map)
        normalized_article_map[self.MAP_KEY__METADATA][self.MAP_KEY__PUBLISHER] = ""
        normalized_article_map[self.MAP_KEY__METADATA][self.MAP_KEY__YEAR] = ""

    def construct_authors_list(self, input_article_map):
        normalized_authors = []
        for author in input_article_map[self.MAP_KEY__CORD19__METADATA][self.MAP_KEY__CORD19__AUTHORS]:
            normalized_author = {
                self.MAP_KEY__FIRST_NAME: author[self.MAP_KEY__CORD19__FIRST_NAME],
                self.MAP_KEY__MIDDLE_NAME: ' '.join(author[self.MAP_KEY__CORD19__MIDDLE_NAME]),
                self.MAP_KEY__LAST_NAME: author[self.MAP_KEY__CORD19__LAST_NAME],
                self.MAP_KEY__AFFILIATION: {
                    self.MAP_KEY__INSTITUTION: ""
                    if self.MAP_KEY__CORD19__INSTITUTION not in author[self.MAP_KEY__CORD19__AFFILIATION]
                    else author[self.MAP_KEY__CORD19__AFFILIATION][self.MAP_KEY__CORD19__INSTITUTION],
                    self.MAP_KEY__ADDRESS: ""
                    if self.MAP_KEY__CORD19__LOCATION not in author[self.MAP_KEY__CORD19__AFFILIATION]
                    else ' '.join(str(x) for x in author[self.MAP_KEY__CORD19__AFFILIATION][
                        self.MAP_KEY__CORD19__LOCATION].values())},
                self.MAP_KEY__EMAIL: author[self.MAP_KEY__CORD19__EMAIL],
            }
            normalized_authors.append(normalized_author)
        return normalized_authors

    def set_paragraphs_list(self, input_article_map, normalized_article_map):
        normalized_paragraphs = []

        # add the info of the abstract paragraphs
        for par in input_article_map[self.MAP_KEY__CORD19__ABSTRACT]:
            normalized_paragraphs.append(self.get_paragraph_info(par))

        # add the body paragraphs info
        for par in input_article_map[self.MAP_KEY__CORD19__BODY_TEXT]:
            normalized_paragraphs.append(self.get_paragraph_info(par))

        normalized_article_map[self.MAP_KEY__PARAGRAPHS] = normalized_paragraphs

    def get_paragraph_info(self, original_paragraph_map):
        normalized_abstract_info = {
            self.MAP_KEY__TEXT: original_paragraph_map[self.MAP_KEY__CORD19__TEXT],
            self.MAP_KEY__SECTION: {
                self.MAP_KEY__TEXT: original_paragraph_map[self.MAP_KEY__CORD19__SECTION],
                self.MAP_KEY__SECTION_DISCOURSE: ""
            },
            self.MAP_KEY__CITATIONS:
                self.construct_citation_list(original_paragraph_map[self.MAP_KEY__CORD19__CITE_SPANS]),
            self.MAP_KEY__ENTITIES:
                self.construct_entities_list(original_paragraph_map[self.MAP_KEY__CORD19__REF_SPANS])
        }
        return normalized_abstract_info

    def construct_citation_list(self, original_citation_list):
        normalized_citation_list = []
        for citation in original_citation_list:
            normalized_citation_list.append({
                self.MAP_KEY__START: citation[self.MAP_KEY__CORD19__START],
                self.MAP_KEY__END: citation[self.MAP_KEY__CORD19__END],
                self.MAP_KEY__TEXT: citation[self.MAP_KEY__CORD19__TEXT],
                self.MAP_KEY__REF_ID: citation[self.MAP_KEY__CORD19__REF_ID],
            })
        return normalized_citation_list

    def construct_entities_list(self, original_entities_list):
        normalized_entities_list = []
        for entity in original_entities_list:
            normalized_entities_list.append({
                self.MAP_KEY__START: entity[self.MAP_KEY__CORD19__START],
                self.MAP_KEY__END: entity[self.MAP_KEY__CORD19__END],
                self.MAP_KEY__TEXT: entity[self.MAP_KEY__CORD19__TEXT],
                self.MAP_KEY__REF_ID: entity[self.MAP_KEY__CORD19__REF_ID],
            })
        return normalized_entities_list
