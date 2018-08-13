import dask as da
import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.sparse import linalg as ln


def calibration_double_ended_calc(self, st_label, ast_label, rst_label,
                                  rast_label):
    nx = 0
    z_list = []
    cal_dts_list = []
    cal_ref_list = []
    x_index_list = []
    for k, v in self.sections.items():
        for vi in v:
            nx += len(self.x.sel(x=vi))

            z_list.append(self.x.sel(x=vi).data)

            # cut out calibration sections
            cal_dts_list.append(self.sel(x=vi))

            # broadcast point measurements to calibration sections
            ref = xr.full_like(cal_dts_list[-1]['TMP'], 1.) * self[k]

            cal_ref_list.append(ref)

            x_index_list.append(
                np.where(
                    np.logical_and(self.x > vi.start, self.x < vi.stop))[
                    0])

    z = np.concatenate(z_list)
    x_index = np.concatenate(x_index_list)

    cal_ref = xr.concat(cal_ref_list, dim='x').data

    chunks_dim = (nx, cal_ref.chunks[1])
    st = da.concatenate(
        [a[st_label].data
         for a in cal_dts_list], axis=0).rechunk(chunks_dim)
    ast = da.concatenate(
        [a[ast_label].data
         for a in cal_dts_list], axis=0).rechunk(chunks_dim)
    rst = da.concatenate(
        [a[rst_label].data
         for a in cal_dts_list], axis=0).rechunk(chunks_dim)
    rast = da.concatenate(
        [a[rast_label].data
         for a in cal_dts_list], axis=0).rechunk(chunks_dim)

    nt = self[st_label].data.shape[1]
    no = self[st_label].data.shape[0]

    p0_est = np.asarray([482., 0.1] + nt * [1.4] + no * [0.])

    # Eqs for F and B temperature
    data1 = np.repeat(1 / (cal_ref.T.ravel() + 273.15), 2)  # gamma
    data2 = np.tile([0., -1.], nt * nx)  # alphaint
    data3 = np.tile([-1., -1.], nt * nx)  # C
    data5 = np.tile([-1., 1.], nt * nx)  # alpha
    # Eqs for alpha
    data6 = np.repeat(-0.5, nt * no)  # alphaint
    data9 = np.ones(nt * no, dtype=float)  # alpha
    data = np.concatenate([data1, data2, data3, data5, data6, data9])

    # (irow, icol)
    coord1row = np.arange(2 * nt * nx, dtype=int)
    coord2row = np.arange(2 * nt * nx, dtype=int)
    coord3row = np.arange(2 * nt * nx, dtype=int)
    coord5row = np.arange(2 * nt * nx, dtype=int)

    coord6row = np.arange(2 * nt * nx, 2 * nt * nx + nt * no, dtype=int)
    coord9row = np.arange(2 * nt * nx, 2 * nt * nx + nt * no, dtype=int)

    coord1col = np.zeros(2 * nt * nx, dtype=int)
    coord2col = np.ones(2 * nt * nx, dtype=int) * (2 + nt + no - 1)
    coord3col = np.repeat(np.arange(nt, dtype=int) + 2, 2 * nx)
    coord5col = np.tile(np.repeat(x_index, 2) + nt + 2, nt)

    coord6col = np.ones(nt * no, dtype=int)
    coord9col = np.tile(np.arange(no, dtype=int) + nt + 2, nt)

    rows = [
        coord1row, coord2row, coord3row, coord5row, coord6row, coord9row
        ]
    cols = [
        coord1col, coord2col, coord3col, coord5col, coord6col, coord9col
        ]
    coords = (np.concatenate(rows), np.concatenate(cols))

    # try scipy.sparse.bsr_matrix
    X = sp.coo_matrix(
        (data, coords),
        shape=(2 * nx * nt + nt * no, nt + 2 + no),
        dtype=float,
        copy=False)
    y1F = da.log(st / ast).T.ravel()
    y1B = da.log(rst / rast).T.ravel()
    y1 = da.stack([y1F, y1B]).T.ravel()
    y2F = np.log(self[st_label].data / self[ast_label].data).T.ravel()
    y2B = np.log(self[rst_label].data / self[rast_label].data).T.ravel()
    y2 = (y2B - y2F) / 2
    y = da.concatenate([y1, y2]).compute()
    p0 = ln.lsqr(X, y, x0=p0_est, show=True, calc_var=True)
    err = (y - X.dot(p0[0]))  # .reshape((nt, nx)).T  # dims: (nx, nt)
    errFW = err[:2 * nt * nx:2].reshape((nt, nx)).T
    errBW = err[1:2 * nt * nx:2].reshape((nt, nx)).T  # dims: (nx, nt)
    ddof = nt + 2 + no
    var_lsqr = p0[-1] * err.std(ddof=ddof) ** 2
    # var_lsqr = ln.inv(X.T.dot(X)).diagonal() * err.std(ddof=ddof) ** 2

    return nt, z, p0, err, errFW, errBW, var_lsqr


def Tvar_double_ended(self, store_IBW_var, store_IFW_var, store_alpha,
                      store_c, store_gamma):
    lF = ('Tvar_' + store_IFW_var,
          'Tvar_' + store_c + 'F',
          'Tvar_' + store_alpha + 'F',
          'Tvar_' + store_gamma + 'F')
    self.attrs['var_composite_label_F'] = lF
    lB = ('Tvar_' + store_IBW_var,
          'Tvar_' + store_c + 'B',
          'Tvar_' + store_alpha + 'B',
          'Tvar_' + store_gamma + 'B')
    self.attrs['var_composite_label_B'] = lB

    factorFW = (self['TMPF'] + 273.15) ** 4 / self[store_gamma] ** 2
    self[lF[0]] = factorFW * self[store_IFW_var]
    self[lF[1]] = factorFW * self[store_c + '_var'].data
    self[lF[2]] = factorFW * self[store_alpha + '_var']
    self[lF[3]] = (self['TMPF'] + 273.15) ** 2 / \
        self[store_gamma] * self[store_gamma + '_var']

    factorBW = (self['TMPB'] + 273.15) ** 4 / self[store_gamma] ** 2
    self[lB[0]] = factorBW * self[store_IBW_var]
    self[lB[1]] = factorBW * self[store_c + '_var'].data
    self[lB[2]] = factorBW * self[store_alpha + '_var']
    self[lB[3]] = (self['TMPB'] + 273.15) ** 2 / \
        self[store_gamma] * self[store_gamma + '_var']
    # self['Tvar_alphaint']

    self['Tvar_F'] = xr.concat(
        objs=[self[li] for li in lF], dim='stacked').sum(dim='stacked')
    self['Tvar_B'] = xr.concat(
        objs=[self[li] for li in lB], dim='stacked').sum(dim='stacked')
    pass
