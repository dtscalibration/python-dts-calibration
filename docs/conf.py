import os
from datetime import date

from xarray import Dataset  # noqa: E401
import sphinx_autosummary_accessors  # noqa: E401

# The following line introduces the .dts accessor for xarray datasets
import dtscalibration  # noqa: E401  # noqa: E401

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_autosummary_accessors",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    # 'matplotlib.sphinxext.mathmpl',
    # 'matplotlib.sphinxext.only_directives',  # not needed after matplotlib
    # >3.0.0
    "matplotlib.sphinxext.plot_directive",
    # 'matplotlib.sphinxext.ipython_directive',
    "recommonmark",  # Parses markdown
]

if os.getenv("SPELLCHECK"):
    extensions += "sphinxcontrib.spelling"
    spelling_show_suggestions = True
    spelling_lang = "en_US"


source_suffix = [".rst", ".md"]
master_doc = "index"
project = "dtscalibration"
year = str(date.today().year)
author = "Bas des Tombe and Bart Schilperoort"
copyright = f"{year}, {author}"
version = release = "3.0.0"

pygments_style = "trac"
templates_path = [".", sphinx_autosummary_accessors.templates_path]
extlinks = {
    "issue": (
        "https://github.com/dtscalibration/python-dts-calibration/issues" "/%s",
        "#",
    ),
    "pr": ("https://github.com/dtscalibration/python-dts-calibration/pull/%s", "PR #"),
}
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

html_theme = "nature"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"],
}
html_short_title = f"{project}-{version}"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

# sphinx_automodapi.automodapi
numpydoc_show_class_members = True

# -- nbsphinx configuration --
nbsphinx_allow_errors = False
nbsphinx_execute = "always"
