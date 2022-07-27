class NormalizerInterface:
    MAP_KEY__DOC_ID = "doc_id"
    MAP_KEY__UID = "uid"
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

    def normalize_input_doc(self, input_path, output_path):
        """
        The input path type will depend on the test collection for example it may be pdf, json ...
        This function is implemented according to the test collection
        """
        pass
