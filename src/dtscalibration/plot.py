# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


def plot_sigma_report(ds, temp_label, temp_var_label, itimes=None):
    assert 'CI' not in ds[temp_label].dims, 'use other plot report function'

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))

    colors = ['#ffeda0', '#feb24c', '#f03b20']

    line_kwargs = dict(linewidth=0.7)

    temp = ds[temp_label].mean(dim='time').compute()
    stds = np.sqrt(ds[temp_var_label]).mean(dim='time')

    for l, c in zip([2., 1.], colors):
        y1 = temp - l * stds
        y2 = temp + l * stds
        label_str = '{0:2.2f}$\sigma$ confidence interval'.format(l)
        ax1.fill_between(y1.x, y1, y2,
                         facecolor=c, label=label_str, alpha=0.9,
                         linewidth=0.7, edgecolor=c)

    if itimes:
        for iitimes in itimes:
            ds[temp_label].isel(time=iitimes).plot(
                ax=ax1, c='grey', label='DTS single', **line_kwargs)

    temp.plot(ax=ax1, linewidth=0.8, c='black', label='DTS avg')

    sigma_est = ds.ufunc_per_section(label=temp_label,
                                     func=np.std,
                                     temp_err=True,
                                     calc_per='stretch')

    for (k, v), (k_se, v_se) in zip(ds.sections.items(),
                                    sigma_est.items()):
        for vi, v_sei in zip(v, v_se):
            val = ds[k].mean(dim='time')
            ax1.plot(
                [vi.start, vi.stop], [val, val],
                linewidth=0.8, c='blue', linestyle='--')
            sig_dts = stds.sel(x=vi).mean()
            tbx, tby = (vi.start + vi.stop) / 2, val
            tbt = "$\sigma_{Est}$ = " + "{0:2.3f}$^\circ$C\n".format(sig_dts.data) + \
                  "$\sigma_{DTS}$ = " + "{0:2.3f}$^\circ$C".format(v_sei)
            ax1.annotate(
                tbt,
                xy=(tbx, tby),
                ha='center',
                fontsize=8,
                xytext=(0, 16),
                textcoords='offset points',
                bbox=dict(fc='white', alpha=0.4, color='none'))

    ax1.set_title('Temperature and standard deviation averaged over '
                  'time per reference section')
    ax1.legend()
    ax1.set_ylabel('Temperature [$^\circ$C]')

    err_ref = ds.ufunc_per_section(label=temp_label,
                                   func=None,
                                   temp_err=True,
                                   calc_per='stretch')
    x_ref = ds.ufunc_per_section(label='x', calc_per='stretch')

    for (k, v), (k_se, v_se), (kx, vx) in zip(ds.sections.items(),
                                              err_ref.items(),
                                              x_ref.items()):
        for vi, v_sei, vxi in zip(v, v_se, vx):
            var_temp_t = np.std(v_sei, axis=1)
            ax2.plot(vxi, var_temp_t, label=k, **line_kwargs)

    stds.plot(ax=ax2, c='black', **line_kwargs)
    ax2.set_ylim([0., 1.05 * stds.max()])
    ax2.set_title('Measured and projected standard deviation averaged over time')
    ax2.legend()
    ax2.set_ylabel('Temperature [$^\circ$C]')

    plt.tight_layout()
