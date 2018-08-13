import glob
import os
import shutil
import subprocess
import tempfile

import nbformat


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    dirname, __ = os.path.split(path)
    os.chdir(dirname)

    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        jupyter_exec = shutil.which('jupyter')

        args = [jupyter_exec, "nbconvert", path,
                "--output", fout.name,
                "--to", "notebook",
                "--execute", "--ExecutePreprocessor.timeout=60"]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]

    return nb, errors


def test_ipynb():
    file_ext = '*.ipynb'
    wd = os.path.dirname(os.path.abspath(__file__))
    nb_dir = os.path.join(wd, '..', 'examples', 'notebooks', file_ext)
    filepathlist = glob.glob(nb_dir)

    for fp in filepathlist:
        nb, errors = _notebook_run(fp)
        assert errors == []

    pass
