import glob
import os
from subprocess import check_call

try:
    # this file is excecuted as script
    wd = os.path.dirname(os.path.realpath(__file__))
    print('run as script: wd', wd)
except:
    # Excecuted from console. pwd = ./docs
    wd = os.getcwd()
    print('run from console: wd', wd)

file_ext = '*.ipynb'
prepend_title_flag = 0

# ./examples/notebooks
inpath = os.path.join(wd, '..', 'examples', 'notebooks')

# ./docs/examples/notebooks
outpath = os.path.join(wd, 'examples', 'notebooks')
fp_index = os.path.join(wd, 'examples', 'index.rst')

filepathlist = sorted(glob.glob(os.path.join(inpath, file_ext)))
filenamelist = [os.path.basename(path) for path in filepathlist]

for fp, fn in zip(filepathlist, filenamelist):
    # save clean notebook to github
    check_call(['jupyter', 'nbconvert',
                '--clear-output',
                '--ClearOutputPreprocessor.enabled=True',
                '--inplace',
                fp])

    # run the notebook to:
    # 1) check whether no errors occur.
    # 2) save and show outputconvert notebook to rst for documentation
    outfilepath = os.path.join(outpath, fn)
    check_call(['jupyter', 'nbconvert',
                # '--execute',
                '--to', 'rst',
                # '--ExecutePreprocessor.kernel_name=python',
                # '--KernelGatewayApp.force_kernel_name=python',
                # "--ExecutePreprocessor.timeout=60",
                '--output', outfilepath,
                fp])


    # prepend title
    if prepend_title_flag:
        title = fn + '\n' + '=' * len(fn) + '\n\n'

        with open(outfilepath + '.rst', 'r') as original:
            data = original.read()
        with open(outfilepath + '.rst', 'w') as modified:
            modified.write(title + data)

# write index file to toc
fp_index = os.path.join(wd, 'examples', 'index.rst')
s = """Learn by Examples
=================

.. toctree::
"""

with open(fp_index, 'w+') as fh:
    fh.write(s)
    for fn in filenamelist:
        sf = "    notebooks/{}.rst".format(fn)
        fh.write(sf)
