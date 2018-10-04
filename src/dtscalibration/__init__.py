# coding=utf-8
from .datastore import DataStore
from .datastore import open_datastore
from .datastore import read_xml_dir
from .datastore import read_xml_list

__version__ = '0.5.0'
__all__ = ["DataStore", "open_datastore", "read_xml_dir", "read_xml_list"]
