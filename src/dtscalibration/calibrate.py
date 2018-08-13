# coding=utf-8
import os

from dtscalibration.datastore import read_xml_dir

filepath = os.path.join('..', '..', 'tests', 'data', 'double_ended2')
ds = read_xml_dir(filepath,
                  timezone_netcdf='UTC',
                  timezone_ultima_xml='Europe/Amsterdam',
                  file_ext='*.xml')
sections = {
    'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
    'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
    }
ds.variance_stokes(st_label='ST', sections=sections)
