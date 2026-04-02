import importlib
import re
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import glide


def test_version_is_set():
    version = glide.__version__
    assert isinstance(version, str)
    assert re.match(r"^\d+\.\d+\.\d+", version)


def test_version_fallback_when_package_not_found():
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        importlib.reload(glide)
        assert glide.__version__ == "unknown"
