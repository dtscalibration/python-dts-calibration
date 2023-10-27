def inverse_variance_weighted_mean(
    self,
    tmp1="tmpf",
    tmp2="tmpb",
    tmp1_var="tmpf_mc_var",
    tmp2_var="tmpb_mc_var",
    tmpw_store="tmpw",
    tmpw_var_store="tmpw_var",
):
    """Compute inverse variance weighted average, and add result in-place.

    Parameters
    ----------
    tmp1 : str
        The label of the first temperature dataset that is averaged
    tmp2 : str
        The label of the second temperature dataset that is averaged
    tmp1_var : str
        The variance of tmp1
    tmp2_var : str
        The variance of tmp2
    tmpw_store : str
        The label of the averaged temperature dataset
    tmpw_var_store : str
        The label of the variance of the averaged temperature dataset

    Returns
    -------

    """

    self[tmpw_var_store] = 1 / (1 / self[tmp1_var] + 1 / self[tmp2_var])

    self[tmpw_store] = (
        self[tmp1] / self[tmp1_var] + self[tmp2] / self[tmp2_var]
    ) * self[tmpw_var_store]

    pass


def inverse_variance_weighted_mean_array(
    self,
    tmp_label="tmpf",
    tmp_var_label="tmpf_mc_var",
    tmpw_store="tmpw",
    tmpw_var_store="tmpw_var",
    dim="time",
):
    """
    Calculates the weighted average across a dimension.

    Parameters
    ----------

    Returns
    -------

    See Also
    --------
    - https://en.wikipedia.org/wiki/Inverse-variance_weighting

    """
    self[tmpw_var_store] = 1 / (1 / self[tmp_var_label]).sum(dim=dim)

    self[tmpw_store] = (self[tmp_label] / self[tmp_var_label]).sum(dim=dim) / (
        1 / self[tmp_var_label]
    ).sum(dim=dim)

    pass
