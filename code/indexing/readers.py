


class CORD19Reader:

    def __init__(self) -> None:
        pass
    

    def read(self, document_data):
        document_obj = {}
        title = document_data["metadata"]["title"]
        abstract = " ".join([p["text"] for p in document_data["paragraphs"] if p["section"]["text"] == "Abstract"])
        text = f"{title}\n{abstract}"
        document_obj["text"] = text
        return document_obj