# kodicare_framework

Kodicare IR framework that calculates Knowledge Delta for test collections

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

Overview of the process
---

### Setting the configuration

Initialize a configuration file to set the paths for the document collection, metadata, working and output. Example of
configuration file `cord19_config.yaml`.

### Normalize the document collection

The goal of this step is to bring the document collection to a known formate to the framework in order to perform
further preprocessing and retrieval experiments. We define a json formate for each document in the collection that the
document should be parsed and converted to it.

So, it's required as a first step to implement the class `preprocessing.normalizer_interface`. An example of this
implementation for the cord19 collection is given in the script `preprocessing.cord19_normalizer`

Example of normalizer json output

```json
{
  "uid": "0a00a6df208e068e7aa369fb94641434ea0e6070",
  "doc_id": "0a00a6df208e068e7aa369fb94641434ea0e6070",
  "cord_uid": "sjcanw19",
  "metadata": {
    "title": "BMC Genomics Novel genome polymorphisms in BCG vaccine strains and impact on efficacy",
    "doc_type": "scientific paper",
    "authors": [
      {
        "first_name": "Andrea",
        "middle_name": "S",
        "last_name": "Leung",
        "affiliation": {
          "institution": "University of Toronto",
          "address": ""
        },
        "email": ""
      },
      ...
    ],
    "publisher": "",
    "year": ""
  },
  "paragraphs": [
    {
      "text": "Bacille Calmette-Gu\u00e9rin (BCG) is an attenuated strain of Mycobacterium bovis currently used as a vaccine against tuberculosis. Global distribution and propagation of BCG has contributed to the in vitro evolution of the vaccine strain and is thought to partially account for the different outcomes of BCG vaccine trials. Previous efforts by several molecular techniques effectively identified large sequence polymorphisms among BCG daughter strains, but lacked the resolution to identify smaller changes. In this study, we have used a NimbleGen tiling array for whole genome comparison of 13 BCG strains. Using this approach, in tandem with DNA resequencing, we have identified six novel large sequence polymorphisms including four deletions and two duplications in specific BCG strains. Moreover, we have uncovered various polymorphisms in the phoP-phoR locus. Importantly, these polymorphisms affect genes encoding established virulence factors including cell wall complex lipids, ESX secretion systems, and the PhoP-PhoR two-component system. Our study demonstrates that major virulence factors are different among BCG strains, which provide molecular mechanisms for important vaccine phenotypes including adverse effect profile, tuberculin reactivity and protective efficacy. These findings have important implications for the development of a new generation of vaccines.",
      "section": {
        "text": "Abstract",
        "discourse_section": ""
      },
      "citations": [],
      "entities": []
    },
    ...
  ]
}
```

### Running Baseline models with Elasticsearch

The baseline models use Elasticsearch for indexing and retrieval. You can install Elasticsearch 7.17 by picking a method
and following the
instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/install-elasticsearch.html).
For this guide, we assume that Elasticsearch is running on `localhost:9200`.

Download the CORD19 data to your `<data_folder>` and store the CORD19 queries as your `<query_file>`.
You can index the CORD19 data to Elasticsearch by using the `/code/scripts/es_index_cord.py` script. This script will
create the Elasticsearch index `<index_name>`. If it already exists, it will be deleted first. To index the title and
the abstract paragraph of the CORD19 files, run the following command from the root directory:

    $ python -m code.scripts.es_index_cord localhost:9200 \
    <index_name> docs ./models/elasticsearch/index_body.json <data_folder>

It is also possible to create an index that splits each CORD19 file into paragraphs and indexes each paragraph as a
separate document:

    $ python -m code.scripts.es_index_cord localhost:9200 \
    <index_name> paragraphs ./models/elasticsearch/index_body_paragraphs.json <data_folder>

You can run the queries by using `/code/scripts/es_rank_cord.py`. This script will load the queries from
the `<query_file>`, run them on the `<index_name>` index and print a list of rankings for each query:

    $ python -m code.scripts.es_rank_cord localhost:9200 \
    <index_name> doc <query_file>

or

    $ python -m code.scripts.es_rank_cord localhost:9200 \
    <index_name> paragraphs <query_file>

### Retrieval using K_d&Delta;

In this retrieval, we calculate Knowledge delta between documents in the same test collection (K_d&Delta;) and based on
the K_d&Delta;, we rerank the results from the base retrieval.

An example of the whole process on the Cord 19 to normalize the documents, calculate the K_d&Delta; and rerank the
results as follows:

1. Normalize the document collection:

```commandline
python -m code.scripts.cord19_normalizer_executer
```

2. Cluster the documents based on similarity

```commandline
python -m code.scripts.cord19_normalized_clusters_executer
```

3. Calculate K_d&Delta; within each cluster

```commandline
python -m code.scripts.cord19_normalizer_document_delta_executer
```

4. Rerank the base retrieval using K_d&Delta;

```commandline
python -m code.scripts.reranking.delta_reranking_cluster_first_executer
```

or by the command

```commandline
python -m code.scripts.reranking.delta_reranking_ignore_cluster_executer
```

5. Generate the qrels for the result

```commandline
python -m code.scripts.qrels_generator
```