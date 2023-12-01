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
    │   ├── analysis                           <- Scripts for analysing data and results 
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
    |   |
    │   └── representations                    <- Scripts to represent text.
    |   |
    │   └── data                    <- Scripts to download or generate data.
    |   |
    │   └── knowledge_delta                    <- Scripts to quantify knowledge delta.
    |   |
    │   └── result_delta                    <- Scripts to quantify results delta.
    |   |
    │   └── kd_rd_prediction                    <- Scripts implement models to predict result delta give knowledge delta.



General Requirements
---

- Python 3.8+

Install required Python packages with pip:

    $ pip install -r requirements.txt

### Setting the configuration

Initialize a configuration file to set the paths for the document collection, evolving collection, metadata, working and output. Example of
configuration file `cord19_config.yaml`.

Process 1 - Predict result delta using knowledge delta on evolving test collections 
---

for this process we expect an Evolving Test Collection which is composed of several test collections. The documents in each collection are represented by a json file as the following example

```json
[
  {
    "id": "ug7v899j",
    "title": "Clinical features of culture-proven Mycoplasma pneumoniae infections at King Abdulaziz University Hospital, Jeddah, Saudi Arabia",
    "contents": "OBJECTIVE: This retrospective chart review describes the epidemiology and clinical features of 40 patients with culture-proven Mycoplasma pneumoniae infections at King Abdulaziz University Hospital, Jeddah, Saudi Arabia. METHODS: Patients with positive M. pneumoniae cultures from respiratory specimens from January 1997 through December 1998 were identified through the Microbiology records. Charts of patients were reviewed. RESULTS: 40 patients were identified, 33 (82.5%) of whom required admission. Most infections (92.5%) were community-acquired. The infection affected all age groups but was most common in infants (32.5%) and pre-school children (22.5%). It occurred year-round but was most common in the fall (35%) and spring (30%). More than three-quarters of patients (77.5%) had comorbidities. Twenty-four isolates (60%) were associated with pneumonia, 14 (35%) with upper respiratory tract infections, and 2 (5%) with bronchiolitis. Cough (82.5%), fever (75%), and malaise (58.8%) were the most common symptoms, and crepitations (60%), and wheezes (40%) were the most common signs. Most patients with pneumonia had crepitations (79.2%) but only 25% had bronchial breathing. Immunocompromised patients were more likely than non-immunocompromised patients to present with pneumonia (8/9 versus 16/31, P = 0.05). Of the 24 patients with pneumonia, 14 (58.3%) had uneventful recovery, 4 (16.7%) recovered following some complications, 3 (12.5%) died because of M pneumoniae infection, and 3 (12.5%) died due to underlying comorbidities. The 3 patients who died of M pneumoniae pneumonia had other comorbidities. CONCLUSION: our results were similar to published data except for the finding that infections were more common in infants and preschool children and that the mortality rate of pneumonia in patients with comorbidities was high."
  },
  {
    "id": "02tnwd4m",
    "title": "Nitric oxide: a pro-inflammatory mediator in lung disease?",
    "contents": "Inflammatory diseases of the respiratory tract are commonly associated with elevated production of nitric oxide (NO\u2022) and increased indices of NO\u2022 -dependent oxidative stress. Although NO\u2022 is known to have anti-microbial, anti-inflammatory and anti-oxidant properties, various lines of evidence support the contribution of NO\u2022 to lung injury in several disease models. On the basis of biochemical evidence, it is often presumed that such NO\u2022 -dependent oxidations are due to the formation of the oxidant peroxynitrite, although alternative mechanisms involving the phagocyte-derived heme proteins myeloperoxidase and eosinophil peroxidase might be operative during conditions of inflammation. Because of the overwhelming literature on NO\u2022 generation and activities in the respiratory tract, it would be beyond the scope of this commentary to review this area comprehensively. Instead, it focuses on recent evidence and concepts of the presumed contribution of NO\u2022 to inflammatory diseases of the lung."
  },
  {
    "id": "ejv2xln0",
    "title": "Surfactant protein-D and pulmonary host defense",
    "contents": "Surfactant protein-D (SP-D) participates in the innate response to inhaled microorganisms and organic antigens, and contributes to immune and inflammatory regulation within the lung. SP-D is synthesized and secreted by alveolar and bronchiolar epithelial cells, but is also expressed by epithelial cells lining various exocrine ducts and the mucosa of the gastrointestinal and genitourinary tracts. SP-D, a collagenous calcium-dependent lectin (or collectin), binds to surface glycoconjugates expressed by a wide variety of microorganisms, and to oligosaccharides associated with the surface of various complex organic antigens. SP-D also specifically interacts with glycoconjugates and other molecules expressed on the surface of macrophages, neutrophils, and lymphocytes. In addition, SP-D binds to specific surfactant-associated lipids and can influence the organization of lipid mixtures containing phosphatidylinositol in vitro. Consistent with these diverse in vitro activities is the observation that SP-D-deficient transgenic mice show abnormal accumulations of surfactant lipids, and respond abnormally to challenge with respiratory viruses and bacterial lipopolysaccharides. The phenotype of macrophages isolated from the lungs of SP-D-deficient mice is altered, and there is circumstantial evidence that abnormal oxidant metabolism and/or increased metalloproteinase expression contributes to the development of emphysema. The expression of SP-D is increased in response to many forms of lung injury, and deficient accumulation of appropriately oligomerized SP-D might contribute to the pathogenesis of a variety of human lung diseases."
  },
...
]
```

a sample of EvTC can be found in the data dir 

### EvEE simulation
you can simulate the creation of an EvEE on a list of time stamped docno

- simulate EvEE by generating list of docids per EE (this generated pkl file of lists that each contains docids per EE)
```commandline
python -m code.scripts.data.evee_simulator_executer --config-name cord19_config config.root_dir='/path/to/data'
```

- to get the documents info per EE for Robust collection 
```commandline
python -m code.scripts.data.robust_dtc_ids__doc__converter --config-name cord19_config config.root_dir='/path/to/data'
```


### TF-IDF KD between document collection

Computes the KD between pairs of document collections using TF-IDF, using the following command

- Generate the vocabulary from EvTC
```commandline
python -m code.scripts.representations.vocab_dictionary_generator --config-name cord19_config config.root_dir='/path/to/data'
```

- Represent each TC in EvTC using TF-IDF
```commandline
python -m code.scripts.representations.bow_tfidf_dtc_doc_collection_representation --config-name cord19_config config.root_dir='/path/to/data'
```

- Generate TF-IDF and calculate KD 
```commandline
python -m code.scripts.knowledge_delta.tfidf_docs_kd_executer --config-name cord19_config config.root_dir='/path/to/data'
```

### RD calculation 
generate labels of 0 or 1 if there is a significant change in the IR system performance between pairs of TCs in EvTC

```commandline
python -m code.scripts.result_delta.binary_rd_labelling_executer --config-name cord19_config config.root_dir='/path/to/data'
```

Note: to execute the above command an evaluation file with the results from indexing each TC is required,
a sample of the file can be accessed on request. 

### KD-RD Prediction 
Creates a feature vector using KD and train a model to predict RD

```commandline
python -m code.scripts.kd_rd_prediction.kd_rd_svm_tfidf --config-name cord19_config config.root_dir='/path/to/data'
```

Process 2 - retrieve by calculating KD between documents 
---

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