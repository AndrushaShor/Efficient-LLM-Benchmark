import json
from google.cloud import storage
from google.oauth2 import service_account
class storage_client():
    def __init__(self, service_account_path, project_id):
        self.project_id = project_id
        self.credentials = service_account.Credentials.from_service_account_file(service_account_path)
        self.client = storage.Client(project=self.project_id,credentials=self.credentials)
    
    # this will overwrite previous versions of blobs
    def upload_blob(self, bucket_name:str, file_path:str, obj_name:str): 
        bucket = self.client.get_bucket(bucket_name)

        obj_name_in_bucket = bucket.blob(obj_name)
        obj_name_in_bucket.upload_from_filename(file_path)

    # used to get any checkpoints of models we had in the past
    def download_blob(self, bucket_name:str, obj_name:str, destination_folder:str, filename):
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(obj_name)

        fp = destination_folder + '/' + filename
        blob.download_to_filename(fp)

    # in case one needs to delete a blob object
    def delete_blob(self, bucket_name:str, obj_name:str):
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(obj_name)
        blob.delete()