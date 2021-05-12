# coding=utf-8
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def plot_residuals_reference_sections(
    resid,
    sections,
    fig=None,
    title=None,
    plot_avg_std=None,
    plot_names=True,
    robust=True,
    units="",
    fig_kwargs=None,
    method="split",
    time_dim="time",
    x_dim="x",
    cmap="RdBu_r",
):
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
    sections : Dict[str, List[slice]]
        The sections obj is normally used to set DataStore.sections, now is
        used toobtain the
        section names to plot the names on top of the residuals.
    fig : Figurehandle, optional
    title : str, optional
        Adds a title to the plot
    plot_names : bool
        Whether the names of the sections are plotted on top of the residuals
    method: str
        'split' will remove the distance between sections to cut down on the
        whitespace.
        'single' will use the previous method, where all sections are in one
        plot.
    time_dim : str
        Name of the time dimension to average/take the variance of
    x_dim : str
        Name of the spatial dimension
    cmap : str
        Matplotlib colormap to use for the residual plot. By default it will
        use a diverging colormap.

    Returns
    -------
    fig : Figurehandle

    """
    if method == "single":
        plot_residuals_reference_sections_single(
            resid,
            fig=fig,
            title=title,
            plot_avg_std=plot_avg_std,
            plot_names=plot_names,
            sections=sections,
            robust=robust,
            units=units,
            fig_kwargs=fig_kwargs,
        )

    elif method != "split":
        raise AssertionError("Unknown method")

    else:
        # Set up the axes with gridspec
        if fig_kwargs is None:
            fig_kwargs = dict()

        if fig is None:
            fig = plt.figure(figsize=(8, 6), **fig_kwargs)

        if title:
            fig.suptitle(title)

        # Create the unsorted list
        section_list = []
        section_name_list = []
        for section in sections:
            for sl in sections[section]:
                section_list.append(sl)
                section_name_list.append(section)

        # Make dictionaries to start sorting
        sections_dict = {}
        section_start_dict = {}
        for ii in range(len(section_list)):
            sections_dict[str(ii)] = [section_name_list[ii], section_list[ii]]
            section_start_dict[str(ii)] = section_list[ii].start

        sorted_sections = sorted(
            sections_dict, key=section_start_dict.__getitem__, reverse=True)

        # Create the sorted name and slice lists
        section_name_list = [
            sections_dict[name][0] for name in sorted_sections]
        section_list = [sections_dict[name][1] for name in sorted_sections]

        resid_sections = [resid.sel(x=section) for section in section_list]

        section_ylims = [[sl.start, sl.stop] for sl in section_list]
        section_height_ratios = [sl.stop - sl.start for sl in section_list]
        nsections = len(section_list)

        grid = plt.GridSpec(
            ncols=3,
            nrows=nsections + 1,
            height_ratios=[sum(section_height_ratios) / 5]
            + section_height_ratios,
            width_ratios=[0.2, 0.8, 0.1],
            hspace=0.15,
            wspace=0.15,
            left=0.08,
            bottom=0.12,
            right=0.9,
            top=0.88)

        _x_ax_avg = fig.add_subplot(grid[0, 1])
        x_ax_avg = _x_ax_avg.twinx()

        cbar_ax = fig.add_subplot(grid[1:, 2])

        section_axes = [0] * nsections
        section_ax_avg = [0] * nsections

        legend_ax = fig.add_subplot(grid[0, 0])

        for ii in range(nsections):
            section_axes[ii] = fig.add_subplot(grid[ii + 1, 1])
            section_ax_avg[ii] = fig.add_subplot(grid[ii + 1, 0])

        # Link all 2dplot axes to their bottom row
        for section_ax in section_axes[:-1]:
            section_axes[-1].get_shared_x_axes().join(
                section_axes[-1], section_ax)
            section_ax.set_xticklabels([])

        # Link all 2dplot axes to their avg axes
        for ii in range(nsections):
            section_ax_avg[ii].get_shared_y_axes().join(
                section_ax_avg[ii], section_axes[ii])
            section_axes[ii].set_yticklabels([])

        # Link all avg axes to their bottom row
        for section_avg in section_ax_avg[:-1]:
            section_ax_avg[-1].get_shared_x_axes().join(
                section_ax_avg[-1], section_avg)
            section_avg.set_xticklabels([])

        # Link the x ax avg to the bottom row
        section_axes[-1].get_shared_x_axes().join(section_axes[-1], x_ax_avg)
        x_ax_avg.set_xticklabels([])

        # Determine vmin, vmax;
        vmin, vmax = resid.quantile([0.02, 0.98])
        maxv = np.max(np.abs([vmin, vmax]))
        vmin = -maxv
        vmax = maxv

        # Normalize the color scale to have 0 be the center
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        # Plot the data
        for ii in range(nsections):
            resid_sections[ii].plot(
                ax=section_axes[ii],
                cbar_ax=cbar_ax,
                cbar_kwargs={"extend": "both"},
                cmap=cmap,
                norm=divnorm,
            )
            section_axes[ii].set_ylabel("")

            resid.sel(x=section_list[ii]).std(dim=time_dim).plot(
                ax=section_ax_avg[ii], y=x_dim, c="blue")
            resid.sel(x=section_list[ii]).mean(dim=time_dim).plot(
                ax=section_ax_avg[ii], y=x_dim, c="orange")
            section_ax_avg[ii].axvline(
                0, linestyle="-", c="black", linewidth=0.8)
            section_ax_avg[ii].set_ylabel("")

        cbar_ax.set_ylabel(units)

        for ii, section_avg in enumerate(section_ax_avg):
            section_avg.set_ylim(section_ylims[ii])
            section_avg.set_xlim([vmin, vmax])
            ticks = section_avg.set_yticks(section_ylims[ii])
            ticks[0].label1.set_verticalalignment("bottom")
            ticks[1].label1.set_verticalalignment("top")
            section_avg.set_xticks(
                [np.floor(vmin * 10) / 10, 0,
                 np.ceil(vmax * 10) / 10])
        section_ax_avg[-1].set_xlabel(units)

        for section_ax in section_axes[:-1]:
            section_ax.set_xlabel("")

        section_ax_avg[np.ceil(nsections / 2).astype(int)
                       - 1].set_ylabel("x (m)")

        # plot the x ax avg
        resid.std(dim=x_dim).plot(ax=x_ax_avg, c="blue")
        resid.mean(dim=x_dim).plot(ax=x_ax_avg, c="orange")
        x_ax_avg.axhline(0, linestyle="-", c="black", linewidth=0.8)
        x_ax_avg.set_xlabel("")
        x_ax_avg.set_ylabel(units)
        _x_ax_avg.set_yticks([])

        # make the legend
        legend_ax.fill_between([], [], facecolor="blue", label="STD")
        legend_ax.fill_between([], [], facecolor="orange", label="MEAN")
        legend_ax.legend(loc="center")
        legend_ax.axis("off")

        # add section names
        if plot_names:
            for ii, section in enumerate(section_list):
                xlim = section_axes[ii].get_xlim()
                xc = (xlim[1] + xlim[0]) / 2
                yc = (section.start + section.stop) / 2
                section_axes[ii].text(
                    xc,
                    yc,
                    s=section_name_list[ii],
                    horizontalalignment="center",
                    verticalalignment="center",
                    bbox=dict(facecolor="white", alpha=0.90, edgecolor="none"),
                )

        return fig


def plot_residuals_reference_sections_single(
    resid,
    fig=None,
    title=None,
    plot_avg_std=None,
    plot_names=True,
    sections=None,
    robust=True,
    units="",
    fig_kwargs=None,
    time_dim="time",
    x_dim="x",
):
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
    time_dim : str
        Name of the time dimension to average/take the variance of
    x_dim : str
        Name of the spatial dimension

    Returns
    -------
    fig : Figurehandle

    """
    if plot_names:
        assert sections is not None, (
            "The sections names are obtained from "
            "the sections dict")

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
    if np.issubdtype(resid[time_dim].dtype, np.float) or np.issubdtype(
            resid[time_dim].dtype, np.int):
        resid.plot.imshow(
            ax=main_ax,
            cbar_ax=cbar_ax,
            cbar_kwargs={"aspect": 10},
            robust=robust)
    else:
        resid.plot(
            ax=main_ax,
            cbar_ax=cbar_ax,
            cbar_kwargs={"aspect": 10},
            robust=robust)
    main_ax.set_yticklabels([])
    main_ax.set_ylabel("")
    cbar_ax.set_ylabel(units)

    # x_ax_avg
    x_ax_avg2 = x_ax_avg.twinx()
    resid.std(dim=x_dim).plot(ax=x_ax_avg2, c="blue")
    resid.mean(dim=x_dim).plot(ax=x_ax_avg2, c="orange")
    x_ax_avg2.axhline(0, linestyle="-", c="black", linewidth=0.8)
    if plot_avg_std is not None:
        x_ax_avg2.axhline(plot_avg_std, linestyle="--", c="blue")

    x_ax_avg.set_xticklabels([])
    x_ax_avg.set_yticklabels([])
    x_ax_avg.set_xlim(main_ax.get_xlim())
    x_ax_avg2.set_ylabel(units)

    # y_ax_avg
    dp = resid.std(dim=time_dim)
    x = dp.values
    y = dp.x
    y_ax_avg.plot(x, y, c="blue")
    dp = resid.mean(dim=time_dim)
    x = dp.values
    y = dp.x
    y_ax_avg.plot(x, y, c="orange")
    y_ax_avg.set_ylim(main_ax.get_ylim())
    y_ax_avg.set_ylabel("x (m)")
    y_ax_avg.set_xlabel(units)
    if plot_avg_std is not None:
        y_ax_avg.axvline(plot_avg_std, linestyle="--", c="blue")

    y_ax_avg.axvline(0, linestyle="-", c="black", linewidth=0.8)
    # reverse axis
    y_ax_avg.set_xlim(y_ax_avg.get_xlim()[::-1])

    # legend
    legend_ax.fill_between([], [], facecolor="blue", label="STD")
    legend_ax.fill_between([], [], facecolor="orange", label="MEAN")
    legend_ax.legend(loc="center")
    legend_ax.axis("off")

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
                    horizontalalignment="center",
                    verticalalignment="center",
                    bbox=dict(facecolor="white", alpha=0.90, edgecolor="none"),
                )

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
    sections=None,
    x_dim="x",
):
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
    x_dim : str
        Name of the spatial dimension

    Returns
    -------
    fig : Figurehandle

    """
    if plot_names:
        assert sections is not None, (
            "The sections names are obtained from "
            "the sections dict")

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
        ax=main_ax, cbar_ax=cbar_ax, cbar_kwargs={"aspect": 20}, robust=True)
    main_ax.set_yticklabels([])
    main_ax.set_ylabel("")
    cbar_ax.set_ylabel(r"$^\circ$C")

    # x_ax_avg
    x_ax_avg2 = x_ax_avg.twinx()
    x_ax_avg2.axhline(0, linestyle="-", c="black", linewidth=0.8)
    if precision_x_avg is not None:
        precision_x_avg.plot(ax=x_ax_avg2, c="blue", linewidth=1.1)
    accuracy_x_avg.plot(ax=x_ax_avg2, c="orange", linewidth=0.9)

    x_ax_avg.set_xticklabels([])
    x_ax_avg.set_yticklabels([])
    x_ax_avg.set_yticks([])
    x_ax_avg.set_xlim(main_ax.get_xlim())
    x_ax_avg2.set_ylabel(r"$^\circ$C")

    # y_ax_avg
    y_ax_avg.axvline(0, linestyle="-", c="black", linewidth=0.8)
    if precision_time_avg is not None:
        x = precision_time_avg.values
        y = precision_time_avg.x
        y_ax_avg.plot(x, y, c="blue", linewidth=1.1)
    x = accuracy_time_avg.values
    y = accuracy_time_avg.x
    y_ax_avg.plot(x, y, c="orange", linewidth=0.9)
    if real_accuracy_time_avg is not None:
        x = real_accuracy_time_avg.values
        y = real_accuracy_time_avg.x
        y_ax_avg.plot(x, y, c="green", linewidth=0.9)

    y_ax_avg.set_ylim(main_ax.get_ylim())
    y_ax_avg.set_xlabel(r"$^\circ$C")
    y_ax_avg.set_ylabel("x (m)")

    # reverse axis
    y_ax_avg.set_xlim(y_ax_avg.get_xlim()[::-1])

    # legend
    if real_accuracy_time_avg is not None:
        legend_ax.fill_between(
            [], [], facecolor="blue", label="Projected precision")
        legend_ax.fill_between(
            [], [], facecolor="orange", label="Projected accuracy")
        legend_ax.fill_between(
            [], [], facecolor="green", label="Measured accuracy")
    else:
        legend_ax.fill_between([], [], facecolor="blue", label="Precision")
        legend_ax.fill_between([], [], facecolor="orange", label="Accuracy")

    legend_ax.legend(loc="right", fontsize=9)
    legend_ax.axis("off")

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
                    horizontalalignment="center",
                    verticalalignment="center",
                    bbox=dict(facecolor="white", alpha=0.90, edgecolor="none"),
                )
                # main_ax.axhline(stretch.start, color='white', linewidth=1.4)
                main_ax.axhline(stretch.start, color="grey", linewidth=0.8)
                # main_ax.axhline(stretch.stop, color='white', linewidth=1.4)
                main_ax.axhline(stretch.stop, color="grey", linewidth=0.8)

                # y-avg-axis
                y_ax_avg.axhline(stretch.start, color="grey", linewidth=0.8)
                y_ax_avg.axhline(stretch.stop, color="grey", linewidth=0.8)

    return fig


def plot_sigma_report(
        ds,
        temp_label,
        temp_var_acc_label,
        temp_var_prec_label=None,
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
    time_dim = ds.get_time_dim(data_var_key=temp_label)
    assert "CI" not in ds[temp_label].dims, "use other plot report function"

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))

    colors = ["#ffeda0", "#feb24c", "#f03b20"]

    line_kwargs = dict(linewidth=0.7)

    if itimes:
        temp = ds[temp_label].isel(time=itimes)
        stds = np.sqrt(ds[temp_var_acc_label].isel(time=itimes)).compute()
    else:
        temp = ds[temp_label].mean(dim=time_dim).compute()
        stds = np.sqrt(ds[temp_var_acc_label]).mean(dim=time_dim).compute()

    for lbl, clr in zip([2.0, 1.0], colors):
        y1 = temp - lbl * stds
        y2 = temp + lbl * stds
        label_str = "{0:2.2f}".format(lbl) + r"$\sigma$ confidence interval"
        ax1.fill_between(
            y1.x,
            y1,
            y2,
            facecolor=clr,
            label=label_str,
            alpha=0.9,
            linewidth=0.7,
            edgecolor=clr,
        )

    if isinstance(itimes, list):
        for iitimes in itimes:
            ds[temp_label].isel(time=iitimes).plot(
                ax=ax1, c="grey", label="DTS single", **line_kwargs)

    temp.plot(ax=ax1, linewidth=0.8, c="black", label="DTS")

    if itimes:
        # std_dts_proj = d.ufunc_per_section(
        #     func=np.std,
        #     label='tmpf_mc_set',
        #     calc_per='stretch',
        #     temp_err=False,
        #     subtract_from_label='tmpf',
        #     axis=[0, 1])
        # std_dts_meas = d.ufunc_per_section(
        #     func=np.std,
        #     label='tmpf',
        #     calc_per='stretch',
        #     temp_err=True,
        #     axis=0)
        sigma_est = ds.ufunc_per_section(
            label=temp_label,
            func=np.std,
            temp_err=True,
            calc_per="stretch",
            axis=0)
    else:
        sigma_est = ds.ufunc_per_section(
            label=temp_label, func=np.std, temp_err=True, calc_per="stretch")

    for (k, v), (k_se, v_se) in zip(ds.sections.items(), sigma_est.items()):
        for vi, v_sei in zip(v, v_se):
            if hasattr(v_sei, "compute"):
                v_sei = v_sei.compute()

            if itimes:
                val = ds[k].mean(dim=time_dim)
            else:
                val = ds[k].isel(time=itimes)

            ax1.plot(
                [vi.start, vi.stop], [val, val],
                linewidth=0.8,
                c="blue",
                linestyle="--")
            sig_dts = stds.sel(x=vi).mean()
            tbx, tby = (vi.start + vi.stop) / 2, val
            tbt = (
                r"$\sigma_{Est}$ = " + "{0:2.3f}".format(sig_dts.data)
                + r"$^\circ$C" + "\n" + r"$\sigma_{DTS}$ = "
                + "{0:2.3f}".format(v_sei) + r"$^\circ$C")
            ax1.annotate(
                tbt,
                xy=(tbx, tby),
                ha="center",
                fontsize=8,
                xytext=(0, 16),
                textcoords="offset points",
                bbox=dict(fc="white", alpha=0.9, color="none"),
            )

    if itimes is None:
        ax1.set_title(
            "Temperature and standard deviation averaged over "
            "time per reference section")
    else:
        ax1.set_title(
            "Projected uncertainty at t={} compared to standard error "
            "in baths".format(itimes))
    ax1.legend()
    ax1.set_ylabel(r"Temperature [$^\circ$C]")

    err_ref = ds.ufunc_per_section(
        label=temp_label, func=None, temp_err=True, calc_per="stretch")
    x_ref = ds.ufunc_per_section(label="x", calc_per="stretch")

    for (k, v), (k_se, v_se), (kx, vx) in zip(ds.sections.items(),
                                              err_ref.items(), x_ref.items()):
        for vi, v_sei, vxi in zip(v, v_se, vx):
            var_temp_t = np.std(v_sei, axis=1)
            ax2.plot(vxi, var_temp_t, label=k, **line_kwargs)

    if temp_var_acc_label:
        stds.plot(ax=ax2, c="black", label="Projected accuracy", **line_kwargs)

    if temp_var_prec_label:
        if itimes:
            stds_prec = np.sqrt(ds[temp_var_prec_label].isel(time=itimes))
        else:
            stds_prec = np.sqrt(ds[temp_var_prec_label]).mean(dim=time_dim)
        stds_prec.plot(
            ax=ax2, c="black", label="Projected precision", **line_kwargs)

    ax2.set_ylim([0.0, 1.1 * stds.max()])
    ax2.legend()
    ax2.set_ylabel(r"Temperature [$^\circ$C]")

    plt.tight_layout()


def plot_location_residuals_double_ended(
        ds, werr, hix, tix, ix_sec, ix_match_not_cal, nt):
    from xarray import Dataset

    nx_sec = ix_sec.size
    npair = hix.size
    nx_match_not_cal = ix_match_not_cal.size
    data_vars = {}
    s = slice(None, nx_sec * nt)
    arr = np.zeros((ds.x.size, nt), dtype=float)
    arr[ix_sec] = werr[s].reshape((nx_sec, nt))  # at ix_sec
    data_vars["werr_F"] = (("x", "time"), arr)
    s = slice(nx_sec * nt, 2 * nx_sec * nt)
    arr = np.zeros((ds.x.size, nt), dtype=float)
    arr[ix_sec] = werr[s].reshape((nx_sec, nt))  # at ix_sec
    data_vars["werr_B"] = (("x", "time"), arr)
    if np.any(hix):
        s = slice(2 * nx_sec * nt, 2 * nx_sec * nt + npair * nt)
        arr = np.zeros((ds.x.size, nt), dtype=float)
        arr[hix] = werr[s].reshape((npair, nt))  # at
        arr[tix] = werr[s].reshape((npair, nt))  # at # ix_sec
        data_vars["werr_eq1"] = (("x", "time"), arr)

        s = slice(
            2 * nx_sec * nt + npair * nt, 2 * nx_sec * nt + 2 * npair * nt)
        arr = np.zeros((ds.x.size, nt), dtype=float)
        arr[hix] = werr[s].reshape((npair, nt))  # at
        arr[tix] = werr[s].reshape((npair, nt))  # at # ix_sec
        data_vars["werr_eq2"] = (("x", "time"), arr)

        s = slice(
            2 * nx_sec * nt + 2 * npair * nt,
            2 * nx_sec * nt + 2 * npair * nt + nx_match_not_cal * nt,
        )
        arr = np.zeros((ds.x.size, nt), dtype=float)
        arr[ix_match_not_cal] = werr[s].reshape(
            (nx_match_not_cal, nt))  # at ix_sec
        data_vars["werr_eq3"] = (("x", "time"), arr)

    dv = Dataset(data_vars=data_vars, coords=dict(x=ds.x, time=ds.time))
    if np.any(hix):
        dv["werr_tot"] = (
            dv["werr_F"]**2 + dv["werr_B"]**2 + dv["werr_eq1"]**2
            + dv["werr_eq2"]**2 + dv["werr_eq3"]**2)
    else:
        dv["werr_tot"] = dv["werr_F"]**2 + dv["werr_B"]**2

    fig, axs = plt.subplots(
        len(dv.data_vars),
        1,
        figsize=(12, 12),
        gridspec_kw=dict(hspace=0.02),
        sharex=True,
        sharey=True,
    )
    vmin, vmax = np.percentile(werr, [5, 95])

    for (k, v), ax in zip(dv.data_vars.items(), axs):
        v.plot(ax=ax, vmin=vmin, vmax=vmax)

    plt.figure()
    dv["werr_tot"].sum(dim="time").plot()
    plt.show()
    return dv
