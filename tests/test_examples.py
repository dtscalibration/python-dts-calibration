import os
from collections import namedtuple
from pathlib import Path
from typing import Optional

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor

NBError = namedtuple("NBError", "title, errname, errvalue, exception")

wd = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(wd, "..", "docs", "notebooks")


@pytest.mark.parametrize(
    "src_path", Path(src_dir).glob("*.ipynb"), ids=lambda x: x.name
)
def test_docs_notebook(src_path):
    print(_test_notebook(src_path, "python3"))


@pytest.mark.xfail
def test_identify_not_working_docs_notebook():
    fp = Path(os.path.join(wd, "data", "docs_notebooks", "01Not_working.ipynb"))
    _test_notebook(fp, "python3")


def _test_notebook(notebook_file, kernel) -> Optional[NBError]:
    """
    Test single notebook.

    Parameters
    ----------
    notebook_file : str
        Source path for notebook
    kernel : str, optional
        Notebook kernel name, by default "python3"

    Returns
    -------
    NBError
        Error tuple

    """
    print(f"\nTesting notebook {notebook_file.name}")
    with open(notebook_file, "rb") as file_handle:
        nb_bytes = file_handle.read()
    nb_text = nb_bytes.decode("utf-8")
    nb_content = nbformat.reads(nb_text, as_version=4)
    exec_processor = ExecutePreprocessor(timeout=600, kernel_name=kernel)

    try:
        print(f"{notebook_file.name} - running notebook: ...")
        exec_processor.preprocess(
            nb_content, {"metadata": {"path": str(notebook_file.parent)}}
        )

    except CellExecutionError as cell_err:
        msg = f"Error executing the notebook '{notebook_file.absolute()}"
        print(msg)
        return NBError(
            f"Error while running notebook {notebook_file}",
            cell_err.ename,
            cell_err.evalue,
            cell_err.args[0],
        )

    return None
