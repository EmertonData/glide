from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("glide-py")
except PackageNotFoundError:
    __version__ = "unknown"
