import logging
import os

import boto3
import botocore
import tqdm

s3 = boto3.resource("s3")
logging.basicConfig(filename='example.log', filemode='w', level=logging.INFO)
log = logging.getLogger("weo-storage-aws")


DEFAULT_SENTINEL2_CACHE_BUCKET: str = "s2-cache"


def exists(object_name, bucket_name=DEFAULT_SENTINEL2_CACHE_BUCKET):
    """Check if object exists in bucket by giving object name"""
    try:
        s3.Object(bucket_name, object_name).load()
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # The object does not exist.
            log.info("Object not found")
            return False
        else:
            # Something else has gone wrong.
            raise
    else:
        # The object does exist.
        log.info(f"Object {object_name} found in {bucket_name}")
        return True


def download(object_name, out_file: str, bucket_name=DEFAULT_SENTINEL2_CACHE_BUCKET):
    """Download object from s3 Bucket to a local file path"""
    log.info(f"Download '{object_name}' from bucket '{bucket_name}' to {out_file}")
    remote_file = s3.Bucket(bucket_name).Object(object_name)
    log.info(f"Starting download for '{remote_file.key}' from bucket '{bucket_name}'")
    download_logger = S3DownloadLogger(remote_file.content_length, remote_file.key)

    s3_client = boto3.client("s3")
    with open(out_file, "wb") as f:
        s3_client.download_fileobj(
            bucket_name, object_name, f, Callback=download_logger
        )

    log.info(f"Finished download for '{remote_file.key}'")


def download_directory(remote_folder_name, bucket_name):
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=remote_folder_name):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key)
        

def download_directory_new(remote_folder_name, bucket_name):
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=remote_folder_name):
        if not obj.key.endswith('/'):  # Ignore directories without files
            local_path = os.path.join(os.getcwd(), obj.key)
            local_dir = os.path.dirname(local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            bucket.download_file(obj.key, local_path)

def download_directory_with_progress(remote_folder_name, bucket_name):
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=remote_folder_name):
        if not obj.key.endswith('/'):  # Ignore directories without files
            local_path = os.path.join(os.getcwd(), obj.key)
            local_dir = os.path.dirname(local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Create a tqdm progress bar with the file size
            with tqdm(total=obj.size, unit='B', unit_scale=True, desc=obj.key) as pbar:
                # Define a function to update the progress bar
                def progress_callback(bytes_amount):
                    pbar.update(bytes_amount)

                # Download the file with the progress callback
                bucket.download_file(obj.key, local_path, Callback=progress_callback)


def upload(file_path, object_name, bucket_name=DEFAULT_SENTINEL2_CACHE_BUCKET):
    """Upload a file to the bucket and assign a file path"""
    data = open(file_path, "rb")
    s3.Bucket(bucket_name).put_object(Key=object_name, Body=data)


def upload_directory(path, folder_path, bucketname):
    """
    Upload all files inside a folder given by path

    Parameters
    ----------
    path : str
        the folder to upload
    folder_path : str
        folder to upload inside the bucket
    bucketname: str
       bucket to upload to
    """
    s3_client = boto3.client("s3")
    for root, dirs, files in os.walk(path):
        for file in files:
            remote_file_path = folder_path + "{}".format(file)
            s3_client.upload_file(
                os.path.join(root, file), bucketname, remote_file_path
            )


def upload_callback(self, size):
    if self.total == 0:
        return
    self.uploaded += size
    log.info("{} %".format(int(self.uploaded / self.total * 100)))


class S3DownloadLogger(object):
    def __init__(self, file_size, filename):
        self._filename = filename
        self._size = file_size
        self._seen_so_far = 0
        self._seen_percentages = dict.fromkeys(
            [5, 10, 25, 50, 75, 90, 95], False
        )  # Define intervals here.

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        percentage = round((self._seen_so_far / self._size) * 100)
        if (
            percentage in self._seen_percentages.keys()
            and not self._seen_percentages[percentage]
        ):
            self._seen_percentages[percentage] = True
            log.info(f"Download progress for '{self._filename}': {percentage}%")