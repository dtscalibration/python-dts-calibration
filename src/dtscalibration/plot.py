# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


def plot_residuals_reference_sections(
        resid,
        fig=None,
        title=None,
        plot_avg_std=None,
        plot_names=True,
        sections=None,
        robust=True,
        units='',
        fig_kwargs=None):
    """
    Analyze the residuals of the reference sections, between the Stokes
    signal and a best-fit
    decaying exponential.

    Parameters
    ----------
    plot_avg_std
    resid : DataArray
        The residuals of the fit to estimate the noise in the measured
        Stokes signal. is returned by `ds.variance_stokes`
    fig : Figurehandle, optional
    title : str, optional
        Adds a title to the plot
    plot_names : bool
        Whether the names of the sections are plotted on top of the residuals
    sections : Dict[str, List[slice]]
        The sections obj is normally used to set DataStore.sections, now is
        used toobtain the
        section names to plot the names on top of the residuals.

    Returns
    -------
    fig : Figurehandle

    """
    if plot_names:
        assert sections is not None, 'The sections names are obtained from ' \
                                     'the sections dict'

    # Set up the axes with gridspec
    if fig_kwargs is None:
        fig_kwargs = dict()

    if fig is None:
        fig = plt.figure(figsize=(8, 6), **fig_kwargs)

    if title:
        fig.suptitle(title)

    grid = plt.GridSpec(
        10,
        10,
        hspace=0.2,
        wspace=0.2,
        left=0.08,
        bottom=0.12,
        right=0.9,
        top=0.88)
    main_ax = fig.add_subplot(grid[2:, 2:-1])
    y_ax_avg = fig.add_subplot(grid[2:, :2])  # xticklabels=[],
    x_ax_avg = fig.add_subplot(grid[:2, 2:-1])  # , sharex=main_ax
    legend_ax = fig.add_subplot(grid[:2, :2], xticklabels=[], yticklabels=[])
    cbar_ax = fig.add_subplot(grid[2:, -1], xticklabels=[], yticklabels=[])
    resid.plot(
        ax=main_ax, cbar_ax=cbar_ax, cbar_kwargs={'aspect': 20}, robust=robust)
    main_ax.set_yticklabels([])
    main_ax.set_ylabel('')
    cbar_ax.set_ylabel(units)

    # x_ax_avg
    x_ax_avg2 = x_ax_avg.twinx()
    resid.std(dim='x').plot(ax=x_ax_avg2, c='blue')
    resid.mean(dim='x').plot(ax=x_ax_avg2, c='orange')
    x_ax_avg2.axhline(0, linestyle='-', c='black', linewidth=0.8)
    if plot_avg_std is not None:
        x_ax_avg2.axhline(plot_avg_std, linestyle='--', c='blue')

    x_ax_avg.set_xticklabels([])
    x_ax_avg.set_yticklabels([])
    x_ax_avg.set_xlim(main_ax.get_xlim())
    x_ax_avg2.set_ylabel(units)

    # y_ax_avg
    dp = resid.std(dim='time')
    x = dp.values
    y = dp.x
    y_ax_avg.plot(x, y, c='blue')
    dp = resid.mean(dim='time')
    x = dp.values
    y = dp.x
    y_ax_avg.plot(x, y, c='orange')
    y_ax_avg.set_ylim(main_ax.get_ylim())
    y_ax_avg.set_ylabel('x (m)')
    y_ax_avg.set_xlabel(units)
    if plot_avg_std is not None:
        y_ax_avg.axvline(plot_avg_std, linestyle='--', c='blue')

    y_ax_avg.axvline(0, linestyle='-', c='black', linewidth=0.8)
    # reverse axis
    y_ax_avg.set_xlim(y_ax_avg.get_xlim()[::-1])

    # legend
    legend_ax.fill_between([], [], facecolor='blue', label='STD')
    legend_ax.fill_between([], [], facecolor='orange', label='MEAN')
    legend_ax.legend(loc='center')
    legend_ax.axis('off')

    if plot_names:
        xlim = main_ax.get_xlim()
        xc = (xlim[1] + xlim[0]) / 2
        for k, section in sections.items():
            for stretch in section:
                yc = (stretch.start + stretch.stop) / 2
                main_ax.text(
                    xc,
                    yc,
                    k,
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.55, edgecolor='none'))

    return fig


def plot_accuracy(
        accuracy,
        accuracy_x_avg,
        accuracy_time_avg,
        precision_x_avg=None,
        precision_time_avg=None,
        real_accuracy_time_avg=None,
        fig=None,
        title=None,
        plot_names=True,
        sections=None):
    """
    Analyze the residuals of the reference sections, between the Stokes
    signal and a best-fit
    decaying exponential.

    Parameters
    ----------
    plot_avg_std
    resid : DataArray
        The residuals of the fit to estimate the noise in the measured
        Stokes signal. is returned by `ds.variance_stokes`
    fig : Figurehandle, optional
    title : str, optional
        Adds a title to the plot
    plot_names : bool, optional
        Whether the names of the sections are plotted on top of the residuals
    sections : Dict[str, List[slice]]
        The sections obj is normally used to set DataStore.sections, now is
        used toobtain the
        section names to plot the names on top of the residuals.

    Returns
    -------
    fig : Figurehandle

    """
    if plot_names:
        assert sections is not None, 'The sections names are obtained from ' \
                                     'the sections dict'

    # Set up the axes with gridspec
    if fig is None:
        fig = plt.figure(figsize=(8, 6))

    if title:
        fig.suptitle(title)

    # setup the axes
    grid = plt.GridSpec(
        10,
        10,
        hspace=0.2,
        wspace=0.2,
        left=0.08,
        bottom=0.12,
        right=0.9,
        top=0.88)
    main_ax = fig.add_subplot(grid[2:, 2:-1])
    y_ax_avg = fig.add_subplot(grid[2:, :2])  # xticklabels=[],
    x_ax_avg = fig.add_subplot(grid[:2, 2:-1])  # , sharex=main_ax
    legend_ax = fig.add_subplot(grid[:2, :2], xticklabels=[], yticklabels=[])
    cbar_ax = fig.add_subplot(grid[2:, -1], xticklabels=[], yticklabels=[])

    accuracy.plot(
        ax=main_ax, cbar_ax=cbar_ax, cbar_kwargs={'aspect': 20}, robust=True)
    main_ax.set_yticklabels([])
    main_ax.set_ylabel('')
    cbar_ax.set_ylabel(r'$^\circ$C')

    # x_ax_avg
    x_ax_avg2 = x_ax_avg.twinx()
    x_ax_avg2.axhline(0, linestyle='-', c='black', linewidth=0.8)
    if precision_x_avg is not None:
        precision_x_avg.plot(ax=x_ax_avg2, c='blue', linewidth=1.1)
    accuracy_x_avg.plot(ax=x_ax_avg2, c='orange', linewidth=0.9)

    x_ax_avg.set_xticklabels([])
    x_ax_avg.set_yticklabels([])
    x_ax_avg.set_yticks([])
    x_ax_avg.set_xlim(main_ax.get_xlim())
    x_ax_avg2.set_ylabel(r'$^\circ$C')

    # y_ax_avg
    y_ax_avg.axvline(0, linestyle='-', c='black', linewidth=0.8)
    if precision_time_avg is not None:
        x = precision_time_avg.values
        y = precision_time_avg.x
        y_ax_avg.plot(x, y, c='blue', linewidth=1.1)
    x = accuracy_time_avg.values
    y = accuracy_time_avg.x
    y_ax_avg.plot(x, y, c='orange', linewidth=0.9)
    if real_accuracy_time_avg is not None:
        x = real_accuracy_time_avg.values
        y = real_accuracy_time_avg.x
        y_ax_avg.plot(x, y, c='green', linewidth=0.9)

    y_ax_avg.set_ylim(main_ax.get_ylim())
    y_ax_avg.set_xlabel(r'$^\circ$C')
    y_ax_avg.set_ylabel('x (m)')

    # reverse axis
    y_ax_avg.set_xlim(y_ax_avg.get_xlim()[::-1])

    # legend
    if real_accuracy_time_avg is not None:
        legend_ax.fill_between(
            [], [], facecolor='blue', label='Projected precision')
        legend_ax.fill_between(
            [], [], facecolor='orange', label='Projected accuracy')
        legend_ax.fill_between(
            [], [], facecolor='green', label='Measured accuracy')
    else:
        legend_ax.fill_between([], [], facecolor='blue', label='Precision')
        legend_ax.fill_between([], [], facecolor='orange', label='Accuracy')

    legend_ax.legend(loc='right', fontsize=9)
    legend_ax.axis('off')

    if plot_names:
        xlim = main_ax.get_xlim()
        xc = (xlim[1] + xlim[0]) / 2
        for k, section in sections.items():
            for stretch in section:
                # main axis
                yc = (stretch.start + stretch.stop) / 2
                main_ax.text(
                    xc,
                    yc,
                    k,
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.35, edgecolor='none'))
                # main_ax.axhline(stretch.start, color='white', linewidth=1.4)
                main_ax.axhline(stretch.start, color='grey', linewidth=0.8)
                # main_ax.axhline(stretch.stop, color='white', linewidth=1.4)
                main_ax.axhline(stretch.stop, color='grey', linewidth=0.8)

                # y-avg-axis
                y_ax_avg.axhline(stretch.start, color='grey', linewidth=0.8)
                y_ax_avg.axhline(stretch.stop, color='grey', linewidth=0.8)

    return fig


def plot_sigma_report(
        ds, temp_label, temp_var_acc_label, temp_var_prec_label=None,
        itimes=None):
    """
    Returns two sub-plots. first a temperature with confidence boundaries.
    Parameters
    ----------
    ds
    temp_label
    temp_var_label
    itimes
    """
    assert 'CI' not in ds[temp_label].dims, 'use other plot report function'

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))

    colors = ['#ffeda0', '#feb24c', '#f03b20']

    line_kwargs = dict(linewidth=0.7)

    if itimes:
        temp = ds[temp_label].isel(time=itimes)
        stds = np.sqrt(ds[temp_var_acc_label].isel(time=itimes)).compute()
    else:
        temp = ds[temp_label].mean(dim='time').compute()
        stds = np.sqrt(ds[temp_var_acc_label]).mean(dim='time').compute()

    for l, c in zip([2., 1.], colors):
        y1 = temp - l * stds
        y2 = temp + l * stds
        label_str = '{0:2.2f}'.format(l) + r'$\sigma$ confidence interval'
        ax1.fill_between(
            y1.x,
            y1,
            y2,
            facecolor=c,
            label=label_str,
            alpha=0.9,
            linewidth=0.7,
            edgecolor=c)

    if isinstance(itimes, list):
        for iitimes in itimes:
            ds[temp_label].isel(time=iitimes).plot(
                ax=ax1, c='grey', label='DTS single', **line_kwargs)

    temp.plot(ax=ax1, linewidth=0.8, c='black', label='DTS')

    if itimes:
        # std_dts_proj = d.ufunc_per_section(
        #     func=np.std,
        #     label='TMPF_MC_set',
        #     calc_per='stretch',
        #     temp_err=False,
        #     subtract_from_label='TMPF',
        #     axis=[0, 1])
        # std_dts_meas = d.ufunc_per_section(
        #     func=np.std,
        #     label='TMPF',
        #     calc_per='stretch',
        #     temp_err=True,
        #     axis=0)
        sigma_est = ds.ufunc_per_section(
            label=temp_label,
            func=np.std,
            temp_err=True,
            calc_per='stretch',
            axis=0)
    else:
        sigma_est = ds.ufunc_per_section(
            label=temp_label, func=np.std, temp_err=True, calc_per='stretch')

    for (k, v), (k_se, v_se) in zip(ds.sections.items(), sigma_est.items()):
        for vi, v_sei in zip(v, v_se):
            if hasattr(v_sei, 'compute'):
                v_sei = v_sei.compute()

            if itimes:
                val = ds[k].mean(dim='time')
            else:
                val = ds[k].isel(time=itimes)

            ax1.plot(
                [vi.start, vi.stop], [val, val],
                linewidth=0.8,
                c='blue',
                linestyle='--')
            sig_dts = stds.sel(x=vi).mean()
            tbx, tby = (vi.start + vi.stop) / 2, val
            tbt = r"$\sigma_{Est}$ = " + "{0:2.3f}".format(
                sig_dts.data) + r"$^\circ$C" + "\n" + \
                r"$\sigma_{DTS}$ = " + "{0:2.3f}".format(
                v_sei) + r"$^\circ$C"
            ax1.annotate(
                tbt,
                xy=(tbx, tby),
                ha='center',
                fontsize=8,
                xytext=(0, 16),
                textcoords='offset points',
                bbox=dict(fc='white', alpha=0.4, color='none'))

    if itimes is None:
        ax1.set_title(
            'Temperature and standard deviation averaged over '
            'time per reference section')
    else:
        ax1.set_title(
            'Projected uncertainty at t={} compared to standard error '
            'in baths'.format(itimes))
    ax1.legend()
    ax1.set_ylabel(r'Temperature [$^\circ$C]')

    err_ref = ds.ufunc_per_section(
        label=temp_label, func=None, temp_err=True, calc_per='stretch')
    x_ref = ds.ufunc_per_section(label='x', calc_per='stretch')

    for (k, v), (k_se, v_se), (kx, vx) in zip(ds.sections.items(),
                                              err_ref.items(), x_ref.items()):
        for vi, v_sei, vxi in zip(v, v_se, vx):
            var_temp_t = np.std(v_sei, axis=1)
            ax2.plot(vxi, var_temp_t, label=k, **line_kwargs)

    if temp_var_acc_label:
        stds.plot(ax=ax2, c='black', label='Projected accuracy', **line_kwargs)

    if temp_var_prec_label:
        if itimes:
            stds_prec = np.sqrt(ds[temp_var_prec_label].isel(time=itimes))
        else:
            stds_prec = np.sqrt(ds[temp_var_prec_label]).mean(dim='time')
        stds_prec.plot(
            ax=ax2, c='black', label='Projected precision', **line_kwargs)

    ax2.set_ylim([0., 1.1 * stds.max()])
    ax2.legend()
    ax2.set_ylabel(r'Temperature [$^\circ$C]')

    plt.tight_layout()
