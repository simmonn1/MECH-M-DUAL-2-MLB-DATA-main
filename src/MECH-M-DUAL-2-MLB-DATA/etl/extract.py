import scipy
import logging
import requests
import io


def extract(url: str, key: str) -> dict:
    """Extract a .mat file from an url"""
    logging.debug(f"Extract data from {url}")
    response = requests.get(url)
    return scipy.io.loadmat(io.BytesIO(response.content))[key]
