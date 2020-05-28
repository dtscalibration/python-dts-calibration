# %%
from src.dtscalibration import read_silixa_files
# from dtscalibration import read_silixa_files
from glob import glob
from time import time
import numpy as np

# %%
files = glob('/media/bart/DTS data 1TB/soil_coil_calibration/*.xml')[:10]

ds = read_silixa_files(files)

ds = ds.sel(x=slice(0, None))

sections = {
    'probe2Temperature': [slice(8, 11), slice(120, 123)],
    'probe1Temperature': [slice(20, 23)]
}
ds.sections = sections

#ds['probe1Temperature'] = ds.probe1Temperature*np.nan
#ds['probe1Temperature'].values = ds.probe1Temperature.values.astype(object)
# %%
ds.calibration_single_ended()

ds.compute()

# %%
np.issubdtype(ds.probe2Temperature.values.dtype, np.floating)
