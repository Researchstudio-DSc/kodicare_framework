import boto3

from code.utils import io_util


class QwantCollectionStatistics:
    def __init__(self, s3_resource: boto3.resource):
        """
        :param s3_resource: the initiated  s3 resource
        """
        self.s3_resource = s3_resource

    def count_collection_documents(self, bucket_name, collection_name):
        kodicare_bucket = self.s3_resource.Bucket(bucket_name)
        collection_objects = kodicare_bucket.objects.filter(
            Prefix=io_util.join(collection_name, 'collection/collector'))
        documents_count = 0
        print("processing collection",  collection_name)
        for collector_obj in collection_objects:
            if collector_obj.key.endswith('.txt'):
                body = collector_obj.get()['Body'].read()
                documents_count += body.count(b'\n')
        return documents_count
