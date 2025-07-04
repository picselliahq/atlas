from picsellia import Client
from picsellia.services.upload.file import FileUploader


class FileService:
    def __init__(self, client: Client):
        self.client = client

    def upload(self, path: str, object_name: str) -> None:
        response = FileUploader(
            self.client.connexion.connector_id,
            self.client.connexion.session,
            self.client.connexion.host,
            self.client.connexion.headers,
        ).upload(object_name, path)[0]
        if not response.status_code != "204":
            print(f"error status is {response.status_code}")
            raise Exception()

        print(f"uploaded on {object_name}")

    def download(self, path: str, object_name: str) -> None:
        url = self.client.connexion.init_download(object_name)
        if not self.client.connexion.do_download_file(
            path, url, is_large=False, force_replace=True
        ):
            print(f"could not download {url}")
            raise Exception()

        print(f"downloaded in {path}")
