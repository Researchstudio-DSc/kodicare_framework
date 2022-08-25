# kodicare_framework

IR framework for Kodicare project


Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                              the creator's initials, and a short `-` delimited description, e.g.
    │                                              `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment, e.g.
    │                                              generated with `pip freeze > requirements.txt`
    │
    ├── code           <- Source code for use in this project.
    │   │
    │   ├── utils                             <- Scripts for utilities funtion 
    │   │
    │   ├── indexing                          <- Scripts to build index the data collection at different stages
    │   │
    │   ├── delta                             <- Scripts to find the deltas between documents
    │   │
    │   |── preprocessing                      <- Scripts to preprocess the documents and queries
    |   |
    │   |── retrieval                          <- Scripts to query indexies and rerank documents
    |   |
    │   |── evaluation                         <- Scripts to evaluate the system performance at different stages
    |   |
    │   └── analysis                           <- Scripts to analyze the evaluation resutls including visulization.



General Requirements
---

- Python 3.8+

Install required Python packages with pip:

    $ pip install -r requirements.txt


Running Baseline models with Elasticsearch
---

The baseline models use Elasticsearch for indexing and retrieval. You can install Elasticsearch 7.17 by picking a method and following the instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/install-elasticsearch.html).
For this guide, we assume that Elasticsearch is running on `localhost:9200`.

Download the CORD19 data to your `<data_folder>` and store the CORD19 queries as your `<query_file>`.
You can index the CORD19 data to Elasticsearch by using the `/code/scripts/es_index_cord.py` script. This script will create the Elasticsearch index `<index_name>`. If it already exists, it will be deleted first. To index the title and the abstract paragraph of the CORD19 files, run the following command from the root directory:

    $ python -m code.scripts.es_index_cord localhost:9200 \
    <index_name> docs ./models/elasticsearch/index_body.json <data_folder>


It is also possible to create an index that splits each CORD19 file into paragraphs and indexes each paragraph as a separate document:

    $ python -m code.scripts.es_index_cord localhost:9200 \
    <index_name> paragraphs ./models/elasticsearch/index_body_paragraphs.json <data_folder>


You can run the queries by using `/code/scripts/es_rank_cord.py`. This script will load the queries from the `<query_file>`, run them on the `<index_name>` index and print a list of rankings for each query:

    $ python -m code.scripts.es_rank_cord localhost:9200 \
    <index_name> doc <query_file>

or 

    $ python -m code.scripts.es_rank_cord localhost:9200 \
    <index_name> paragraphs <query_file>