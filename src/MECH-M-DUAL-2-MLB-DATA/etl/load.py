import scipy
import numpy.typing as npt
from pathlib import Path
from .extract import extract
from .transform import transform


url_prefix = ("https://github.com/dynamicslab/databook_python/"
              "raw/refs/heads/master/DATA/")


def load(filename: str) -> npt.NDArray:
    """Load .mat file from the `data` directory and return the consisting of
    the prefix of the name and _wave"""
    dir = Path("data")
    dir.mkdir(parents=True, exist_ok=True)

    file = dir / filename
    prefix = filename.split("Data")[0]
    if not file.exists():
        url = url_prefix + filename.split("_")[0] + ".mat"
        images = extract(url, prefix)
        images_w = transform(images)
        scipy.io.savemat(file, {prefix + "_wave": images_w})

    return scipy.io.loadmat(file)[prefix + "_wave"]
