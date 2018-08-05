# coding=utf-8
import xarray as xr
import yaml


class DataStore(xr.Dataset):
    def __init__(self, *args, **kwargs):
        super(DataStore, self).__init__(*args, **kwargs)

        if '_sections' not in self.attrs:
            self.attrs['_sections'] = yaml.dump(None)

        if 'sections' in kwargs:
            self.sections = kwargs['sections']

    @property
    def sections(self):
        assert hasattr(self, '_sections')
        return yaml.load(self.attrs['_sections'])

    @sections.setter
    def sections(self, sections):
        # assert sections
        self.attrs['_sections'] = yaml.dump(sections)
        pass

    @sections.deleter
    def sections(self):
        self.sections = None
        pass


def open_datastore(path, **kwargs):
    ds_xr = xr.open_dataset(path)
    ds = DataStore(data_vars=ds_xr.data_vars,
                   coords=ds_xr.coords,
                   attrs=ds_xr.attrs,
                   **kwargs)
    return ds
