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

    # Create a temporary file to write the notebook to.
    # 'with' method is used so the file is closed by tempfile
    # and free to be overwritten.
    # with tempfile.NamedTemporaryFile('w', suffix=".ipynb") as fout:
    with tempfile.NamedTemporaryFile('w', suffix=".nbconvert.ipynb", delete=False) as fout:
        nbpath = fout.name

        jupyter_exec = shutil.which('jupyter')

        # recent version (~7.3.1) requires output without extension
        out_path = os.path.join(
            os.path.dirname(nbpath),
            os.path.basename(nbpath).split('.', 1)[0])
        args = [
            jupyter_exec, "nbconvert", path, "--output", out_path, "--to",
            "notebook", "--execute", "--ExecutePreprocessor.timeout=60"]
        subprocess.check_call(args)

        assert os.path.exists(nbpath), "nbconvert used different output filename"

        nb = nbformat.read(nbpath, nbformat.current_nbformat)

    errors = [
        output for cell in nb.cells if "outputs" in cell
        for output in cell["outputs"] if output.output_type == "error"]

    # Remove the temp file once the test is done
    if os.path.exists(nbpath):
        os.remove(nbpath)

    return nb, errors


def test_ipynb():
    file_ext = '*.ipynb'
    wd = os.path.dirname(os.path.abspath(__file__))
    nb_dir = os.path.join(wd, '..', 'examples', 'notebooks', file_ext)
    filepathlist = glob.glob(nb_dir)

    for fp in filepathlist:
        _, errors = _notebook_run(fp)
        assert errors == []
