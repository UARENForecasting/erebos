import numpy as np
import xarray as xr


from erebos import __version__
from erebos.adapters.calipso import process_calipso_dataset
from erebos.adapters.goes import process_goes_dataset
from erebos.utils import RotatedECRPosition


def get_processing_func(ds):
    for val in ds.attrs.values():
        if "goes" in val.lower():
            return process_goes_dataset
        elif "calipso" in val.lower():
            return process_calipso_dataset
    raise TypeError("Not a GOES or Calipso dataset")


def construct_tree(ds):
    from pykdtree.kdtree import KDTree

    X, Y = np.meshgrid(ds.x.values, ds.y.values)
    pts = np.asarray([X.reshape(-1), Y.reshape(-1)]).T
    tree = KDTree(pts)
    return tree


@xr.register_dataset_accessor("erebos")
class ErebosDataset:
    def __init__(self, xarray_obj):
        func = get_processing_func(xarray_obj)
        self._xarray_obj = func(xarray_obj)

    def __getattr__(self, name):
        return getattr(self._xarray_obj, name)

    def __repr__(self):
        return repr(self._xarray_obj)

    @property
    def x(self):
        try:
            return self._xarray_obj.erebos_x
        except AttributeError:
            return self._xarray_obj.x

    @property
    def y(self):
        try:
            return self._xarray_obj.erebos_y
        except AttributeError:
            return self._xarray_obj.y

    @property
    def mean_time(self):
        if "erebos_mean_time" in self._xarray_obj.attrs:
            return self._xarray_obj.attrs["erebos_mean_time"]
        raise AttributeError()

    @property
    def crs(self):
        if "erebos_crs" in self._xarray_obj.coords:
            return self._xarray_obj.coords["erebos_crs"].item()
        raise AttributeError("crs attribute is not available")

    @property
    def spacecraft_location(self):
        if "erebos_spacecraft_location" in self._xarray_obj.coords:
            o = self._xarray_obj.coords["erebos_spacecraft_location"].values
            return RotatedECRPosition(*o)
        raise AttributeError("spacecraft_location attribute is not available")

    @property
    def kdtree(self):
        if not hasattr(self, "_kdtree"):
            self._kdtree = construct_tree(self._xarray_obj)
        return self._kdtree

    def to_netcdf(self, path, engine="h5netcdf", **kwargs):
        ds = self._xarray_obj
        keys = list(ds.data_vars) + list(ds.coords)
        ds = ds.drop([k for k in keys if k.startswith("erebos")])
        for attr in list(ds.attrs.keys()):
            if attr.startswith("erebos"):
                del ds.attrs[attr]
        ds.attrs["erebos_version"] = __version__
        ds.to_netcdf(path, engine=engine, **kwargs)
