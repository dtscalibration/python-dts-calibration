from dtscalibration.datastore_utils import check_dims
from dtscalibration.datastore_utils import get_netcdf_encoding
from dtscalibration.datastore_utils import merge_double_ended
from dtscalibration.datastore_utils import shift_double_ended
from dtscalibration.datastore_utils import suggest_cable_shift_double_ended
from dtscalibration.io import read_apsensing_files
from dtscalibration.io import read_sensornet_files
from dtscalibration.io import read_sensortran_files
from dtscalibration.io import read_silixa_files
from dtscalibration.plot import plot_accuracy
from dtscalibration.plot import plot_location_residuals_double_ended
from dtscalibration.plot import plot_residuals_reference_sections
from dtscalibration.plot import plot_residuals_reference_sections_single
from dtscalibration.plot import plot_sigma_report

__version__ = "2.0.0"
__all__ = [
    "read_apsensing_files",
    "read_sensornet_files",
    "read_sensortran_files",
    "read_silixa_files",
    "check_dims",
    "get_netcdf_encoding",
    "merge_double_ended",
    "shift_double_ended",
    "suggest_cable_shift_double_ended",
    "plot_accuracy",
    "plot_location_residuals_double_ended",
    "plot_residuals_reference_sections",
    "plot_residuals_reference_sections_single",
    "plot_sigma_report",
]

# filenames = ['datastore.py', 'datastore_utils.py', 'calibrate_utils.py',
#              'plot.py', 'io_utils.py']
# filenames = ['plot.py']
#
# for filename in filenames:
#     with open(join(dirname(__file__), filename)) as file:
#         node = ast.parse(file.read())
#
#     functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
#     classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
#     __all__.extend([i.name for i in functions])
#
# __all__.sort()
# print(__all__)
