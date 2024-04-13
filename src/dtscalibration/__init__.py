import importlib.metadata
import warnings

from dtscalibration.dts_accessor import DtsAccessor  # noqa: F401
from dtscalibration.dts_accessor_utils import get_netcdf_encoding
from dtscalibration.dts_accessor_utils import merge_double_ended
from dtscalibration.dts_accessor_utils import shift_double_ended
from dtscalibration.dts_accessor_utils import suggest_cable_shift_double_ended
from dtscalibration.io.apsensing import read_apsensing_files
from dtscalibration.io.sensornet import read_sensornet_files
from dtscalibration.io.sensortran import read_sensortran_files
from dtscalibration.io.silixa import read_silixa_files

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "DtsAccessor",
    "read_apsensing_files",
    "read_sensornet_files",
    "read_sensortran_files",
    "read_silixa_files",
    "get_netcdf_encoding",
    "merge_double_ended",
    "shift_double_ended",
    "suggest_cable_shift_double_ended",
]

warnings.filterwarnings(
    "ignore",
    message="Converting non-nanosecond precision timedelta values to nanosecond",
)
