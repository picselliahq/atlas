from picsellia import Client


def get_client(host: str, api_token: str, organization_id: str):
    client = Client(
        api_token=api_token,
        organization_id=organization_id,
        host=host,
    )
    return client
