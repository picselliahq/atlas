import os

from picsellia import Client
from picsellia.services.upload.file import FileUploader


def get_object_name(
    organization_id: str, dataset_version_id: str, filename: str
) -> str:
    return f"{organization_id}/datasets/versions/{dataset_version_id}/{filename}"


def upload(client: Client, dataset_version_id: str, path: str) -> str:
    filename = os.path.basename(path)
    object_name = get_object_name(
        client.connexion.organization_id, dataset_version_id, filename
    )
    response = FileUploader(
        client.connexion.connector_id,
        client.connexion.session,
        client.connexion.host,
        client.connexion.headers,
    ).upload(object_name, path)[0]
    if not response.status_code != "204":
        print(f"error status is {response.status_code}")
        raise Exception()

    print(f"uploaded on {object_name}")
    return object_name


def download(client: Client, dataset_version_id: str, path: str):
    filename = os.path.basename(path)
    object_name = get_object_name(
        client.connexion.organization_id, dataset_version_id, filename
    )
    url = client.connexion.init_download(object_name)
    if not client.connexion.do_download_file(
        path, url, is_large=False, force_replace=True
    ):
        print(f"could not download {url}")
        raise Exception()

    print(f"downloaded in {path}")


if __name__ == "__main__":
    host = "http://localhost:8000"
    api_token = "<TODO>"
    client = Client(api_token=api_token, host=host)
    dataset_version_id = "0192ba37-0415-75fd-ad27-451d06359a52"

    path = "/home/thomas/dev/playground/playground/0191d5a2-00eb-7d2b-b776-5b8f2e8266d8-logs.json"
    upload(client, dataset_version_id, path)
    os.remove(path)
    download(client, dataset_version_id, path)
