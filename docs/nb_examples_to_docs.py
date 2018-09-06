# coding=utf-8
import glob
import os
import shutil
from subprocess import check_call

clean_nb = False  # save clean notebook to github

try:
    # this file is excecuted as script
    wd = os.path.dirname(os.path.realpath(__file__))
    print('run as script: wd', wd)
except:
    # Excecuted from console. pwd = ./docs
    wd = os.getcwd()
    print('run from console: wd', wd)
    pass

file_ext = '*.ipynb'

# ./examples/notebooks
inpath = os.path.join(wd, '..', 'examples', 'notebooks')

# ./docs/examples/notebooks
outpath = os.path.join(wd, 'examples', 'notebooks')
fp_index = os.path.join(wd, 'examples', 'index.rst')

# clean outputdir
shutil.rmtree(outpath)
os.makedirs(outpath)

filepathlist = sorted(glob.glob(os.path.join(inpath, file_ext)))
filenamelist = [os.path.basename(path) for path in filepathlist]

for fp, fn in zip(filepathlist, filenamelist):
    if clean_nb:
        # save clean notebook to github
        check_call(['jupyter', 'nbconvert',
                    '--clear-output',
                    '--ClearOutputPreprocessor.enabled=True',
                    '--inplace',
                    fp])
    else:
        check_call(['jupyter', 'nbconvert',
                    '--execute',
                    '--ExecutePreprocessor.kernel_name=python',
                    '--KernelGatewayApp.force_kernel_name=python',
                    "--ExecutePreprocessor.timeout=60",
                    '--inplace',
                    fp])
    # run the notebook to:
    # 1) check whether no errors occur.
    # 2) save and show outputconvert notebook to rst for documentation
    # outfilepath = os.path.join(outpath, fn)
    check_call(['jupyter', 'nbconvert',
                '--execute',
                '--to', 'rst',
                '--ExecutePreprocessor.kernel_name=python',
                '--KernelGatewayApp.force_kernel_name=python',
                "--ExecutePreprocessor.timeout=60",
                '--output-dir', outpath,
                '--output', fn,
                fp])

# write index file to toc
fp_index = os.path.join(wd, 'examples', 'index.rst')
s = """Learn by Examples
=================

.. toctree::
"""

with open(fp_index, 'w+') as fh:
    fh.write(s)
    for fn in filenamelist:
        sf = "    notebooks/{}.rst\n".format(fn)
        fh.write(sf)
