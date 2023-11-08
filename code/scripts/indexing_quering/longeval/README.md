# Indexing and retrieval longeval epoch

---
## Indexing 
- To index a long eval epoch use the following command from the root directory of the project

```commandline
python -m code.scripts.indexing_quering.longeval.pyterrier_index_creation_executer --config-name longeval_config test_collection.documents_dir='/path/to/doc/dir' index.index_path='/path/to/store/index'
```

- To index a long eval epoch use the following command from the root directory of the project

```commandline
python -m code.scripts.indexing_quering.longeval.pyterrier_retrieval_executer --config-name longeval_config test_collection.queries_path=/path/to/queries/tsv/file test_collection.qrels_path=/path/to/qrels/txt/file index.index_path=/path/of/stored/index evaluation.evaluation_output_path=/path/to/store/evaluation/result/tsv/file
```
