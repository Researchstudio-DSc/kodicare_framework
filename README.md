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
