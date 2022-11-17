import hydra
import boto3
from code.analysis.data import qwant_collection_statistics


@hydra.main(version_base=None, config_path="../../conf", config_name="qwant_config")
def main(cfg):
    # initiate the s3 resource
    s3_resource = boto3.resource(service_name='s3',
                                 region_name=cfg.s3_config.region,
                                 aws_access_key_id=cfg.s3_config.aws_access_key_id,
                                 aws_secret_access_key=cfg.s3_config.aws_secret_access_key,
                                 endpoint_url=cfg.s3_config.endpoint_url)

    qwant_statistics = qwant_collection_statistics.QwantCollectionStatistics(s3_resource)

    for collection in cfg.collections:
        print(collection)
        print('#documents:', qwant_statistics.count_collection_documents('kodicare', collection))
        print('#queries:', qwant_statistics.count_collection_queries('kodicare', collection))


if __name__ == '__main__':
    main()
