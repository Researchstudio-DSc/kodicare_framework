class NormalizerInterface:
    MAP_KEY__DOC_ID = "doc_id"
    MAP_KEY__UID = "uid"
    MAP_KEY__METADATA = "metadata"
    MAP_KEY__TITLE = "title"
    MAP_KEY__DOC_TYPE = "doc_type"
    MAP_KEY__AUTHORS = "authors"
    MAP_KEY__FIRST_NAME = "first_name"
    MAP_KEY__MIDDLE_NAME = "middle_name"
    MAP_KEY__LAST_NAME = "last_name"
    MAP_KEY__AFFILIATION = "affiliation"
    MAP_KEY__ADDRESS = "address"
    MAP_KEY__INSTITUTION = "institution"
    MAP_KEY__EMAIL = "email"
    MAP_KEY__PUBLISHER = "publisher"
    MAP_KEY__YEAR = "year"
    MAP_KEY__PARAGRAPHS = "paragraphs"
    MAP_KEY__CITATIONS = "citations"
    MAP_KEY__START = "start"
    MAP_KEY__END = "end"
    MAP_KEY__ENTITIES = "entities"
    MAP_KEY__REF_ID = "ref_id"
    MAP_KEY__SECTION = "section"
    MAP_KEY__SECTION_DISCOURSE = "discourse_section"
    MAP_KEY__TEXT = "text"

    DOC_TYPE_SCIENTIFIC_PAPER = "scientific paper"

    def normalize_input_doc(self, input_path, output_path, metadata=None):
        """
        The input path type will depend on the test collection for example it may be pdf, json ...
        This function is implemented according to the test collection
        """
        pass
