# coding=utf-8
from .datastore import DataStore
from .datastore import open_datastore
from .datastore import plot_dask
from .datastore import read_silixa_files

__version__ = '0.5.1'
__all__ = ["DataStore", "open_datastore", "read_silixa_files", "plot_dask"]
