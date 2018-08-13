# coding=utf-8
import os

from dtscalibration.datastore import DataStore
from dtscalibration.datastore import read_xml_dir

filepath = os.path.join('..', '..', 'tests', 'data')
ds = read_xml_dir(filepath,
                  timezone_netcdf='UTC',
                  timezone_ultima_xml='Europe/Amsterdam',
                  file_ext='*.xml')

ds = DataStore()
sections1 = {
    'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
    'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
    }
sections2 = {
    'probe1Temperature': [slice(0., 17.), slice(70., 80.)],  # cold bath
    'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
    }
ds.sections = sections1

assert isinstance(ds._sections, str)

assert ds.sections == sections1
assert ds.sections != sections2

# delete property
del ds.sections
assert ds.sections is None

pass
