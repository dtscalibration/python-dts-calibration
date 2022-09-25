#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')).read()


def get_authors(file='AUTHORS.rst'):
    auth1 = read(file).split('*')[1:]

    auth2 = []
    for ai in auth1:
        auth2.append(ai.split('-')[0].strip())

    return ', '.join(auth2)


setup(
    name='dtscalibration',
    version='1.1.0',
    license='BSD 3-Clause License',
    description='A Python package to load raw DTS files, perform a '
    'calibration, and plot the result',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub(
            '', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))),
    long_description_content_type='text/x-rst',
    author=get_authors(file='AUTHORS.rst'),
    author_email='bdestombe@gmail.com',
    url='https://github.com/dtscalibration/python-dts-calibration',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    python_requires='>= 3.6',  # default dictionary is sorted
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities'],
    keywords=[
        'DTS',
        'Calibration'],
    install_requires=[
        'numpy',
        'xarray<=2022.03.0',
        'pyyaml',
        'xmltodict',
        'scipy',
        'patsy',  # a dependency of statsmodels
        'statsmodels',
        'nbsphinx',
        'dask',
        'toolz',
        'matplotlib',
        'netCDF4<=1.5.8',
        'pandas>=0.24.1',
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
        # https://github.com/pydata/xarray/issues/3401 cloud pickle should be
        # possible to remove from requirements soon. 2019/10/31
        'cloudpickle',
    ],
    extras_require={
        'dev': [
            'bump2version',
            'coverage [toml]',
            'isort',
            'mypy',
            'myst_parser',
            'prospector[with_pyroma]',
            'pytest',
            'pytest-cov',
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx-autoapi',
            'tox',
            'jupyter',
            'nbformat',  # Needed to run the tests
            'pypandoc'
        ],
        'docs': [
            'IPython',
            'nbsphinx',
            'recommonmark',
            'sphinx<6',
            'sphinx-automodapi',
            'pypandoc',
            'jupyter_client',
            'ipykernel',
        ],
        'publishing': [
            'twine',
            'wheel',
        ],
    },
    entry_points={
        'console_scripts': ['dtscalibration = dtscalibration.cli:main']},
)
