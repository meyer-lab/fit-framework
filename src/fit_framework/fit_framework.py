from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def fit(
    opt_df: pd.DataFrame,
    var_params: list[str],
    stratifiers: Dict[str, str],
    param_bounds: Dict[str, tuple[float, float]],
    infer_signal_fn: Callable[[pd.DataFrame], np.ndarray],
) -> pd.DataFrame:
    """
    Fits the model to the data using least squares optimization.

    Args:
        opt_df: DataFrame containing the data to fit.
        var_params: list of variable parameter names.
        stratifiers: Dictionary mapping parameters to their stratification columns.
        param_bounds: Dictionary of parameter bounds.
        infer_signal_fn: Function to infer the signal from the data.

    Returns:
        DataFrame with optimized parameters and inferred signals.
    """
    var_vals, const_vals = marshal_df(opt_df, var_params, stratifiers)
    x0 = np.concatenate(var_vals)
    args = (const_vals, var_params, stratifiers, infer_signal_fn)

    resid_init = residuals(x0, *args)

    assert "signal" in opt_df.columns, "Signal column not found"

    # assert no stratifiers are in var_params
    assert not set(var_params).intersection(
        set(stratifiers.values())
    ), "Stratifiers should not be in var_params"

    for s_param in stratifiers:
        assert s_param in var_params, f"{s_param} (in stratifiers) is not in var_params"

    for b_param in param_bounds:
        assert (
            b_param in var_params
        ), f"{b_param} (in param_bounds) is not in var_params"

    print(
        f"Fitting {resid_init.size} points with "
        f"{np.concatenate(var_vals).size} parameters."
    )

    result = least_squares(
        residuals,
        x0,
        args=args,
        bounds=assemble_bounds_arrs(param_bounds, var_params, const_vals, stratifiers),
        jac_sparsity=assemble_jac_sparsity(var_params, const_vals, stratifiers),
        x_scale="jac",  # type: ignore
        jac="2-point",
        verbose=2,
        max_nfev=50,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-3,
    )

    var_vals_opt = separate_var_vals(result.x, const_vals, var_params, stratifiers)

    result_df = unmarshal_df(var_vals_opt, const_vals, var_params, stratifiers)

    # populate result_df with inferred signal
    result_df["signal_inferred"] = infer_signal_fn(result_df)
    result_df["se"] = np.square(result_df["signal"] - result_df["signal_inferred"])

    return result_df


def residuals(
    var_vals: np.ndarray,
    const_vals: pd.DataFrame,
    var_params: list[str],
    stratifiers: Dict[str, str],
    infer_signal_fn: Callable[[pd.DataFrame], np.ndarray],
) -> np.ndarray:
    """
    Calculates residuals between inferred and actual signals.

    Args:
        var_vals: Array of variable parameter values.
        const_vals: DataFrame of constant parameter values.
        var_params: List of variable parameter names.
        stratifiers: Dictionary mapping parameters to their stratification
            columns.
        infer_signal_fn: Function to infer the signal from the data.

    Returns:
        Array of residuals.
    """
    var_vals_separate = separate_var_vals(var_vals, const_vals, var_params, stratifiers)
    df = unmarshal_df(
        var_vals_separate,
        const_vals,
        var_params,
        stratifiers,
    )
    return infer_signal_fn(df) - df["signal"].values


def multiplex(
    vals: np.ndarray, col_name: str, stratifier: str | None, target_df: pd.DataFrame
) -> None:
    """
    Assigns values to a DataFrame column based on stratification.

    This function populates a column in the target DataFrame with values from
    the input array.  If a stratifier is provided, it assigns one value per
    group defined by each unique value of the stratifier.  If no stratifier is
    given, it assigns the same value to all rows.

    Args:
        vals: Array of values to assign. Its length should match the number of
            unique stratifier values (or be 1 if no stratifier is used).
        col_name: Name of the column in target_df to populate with values.
        stratifier: Name of the column in target_df used for stratification. If
            None, no stratification is used.
        target_df: Target DataFrame to modify. The function operates on this
            DataFrame in-place.

    Returns:
        None
    """
    if stratifier is None:
        target_df[col_name] = vals[0]
    else:
        # this only works consistently because unique() returns values in the
        # order they appear
        for stratifier_val, val in zip(target_df[stratifier].unique(), vals):
            target_df.loc[target_df[stratifier] == stratifier_val, col_name] = val


def demultiplex(
    df: pd.DataFrame, col_name: str, stratifier: Optional[str]
) -> np.ndarray:
    """
    Extracts unique values from a DataFrame column based on stratification.

    This function retrieves unique values from a specified column in the
    DataFrame.  If a stratifier is provided, it extracts one value for each
    group defined by the unique values of the stratifier.  If no stratifier is
    given, it expects the column to have a single unique value across all rows.

    Args:
        df: Source DataFrame containing the data.
        col_name: Name of the column from which to extract values.
        stratifier: Name of the column used for stratification. If None, no
            stratification is used.

    Returns:
        Array of unique values. The length of the array will be equal to the
        number of unique stratifier values (or 1 if no stratifier is used).

    Raises:
        AssertionError: If the stratification is not unique (i.e., multiple
            values found within a group).
    """
    if stratifier is None:
        valss = df[col_name].unique()
        assert len(valss) == 1, "Stratification is not unique"
    else:
        valss = np.array([])
        for _, group in df.groupby(stratifier, sort=False):
            vals = group[col_name].unique()
            assert len(vals) == 1, "Stratification is not unique"
            valss = np.append(valss, vals[0])
    return np.array(valss)


def marshal_df(
    df: pd.DataFrame,
    var_params: list[str],
    stratifiers: Dict[str, str],
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """
    Separates variable and constant parameters from a DataFrame into a format
    suitable for optimization.

    This function extracts variable parameters as lists of arrays (one array per
    parameter) and constant parameters as a DataFrame. Variable parameters can be
    stratified, resulting in multiple values per parameter.

    Args:
        df: Source DataFrame containing all parameters.
        var_params: List of column names in df corresponding to variable parameters.
        stratifiers: Dictionary mapping variable parameter names to their
            stratification columns. If a parameter is not stratified, it should
            not be included in this dictionary.

    Returns:
        A tuple containing:
        - List of numpy arrays, each containing values for one variable parameter.
          Stratified parameters will have multiple values.
        - DataFrame containing all constant parameters.
    """
    var_vals = [demultiplex(df, v, stratifiers.get(v)) for v in var_params]
    const_params = [col for col in df.columns if col not in var_params]
    const_vals = df[const_params]
    return var_vals, const_vals


def unmarshal_df(
    var_vals: list[np.ndarray],
    const_vals: pd.DataFrame,
    var_params: list[str],
    stratifiers: Dict[str, str],
) -> pd.DataFrame:
    """
    Combines variable and constant parameters into a single DataFrame.

    This function is essentially the inverse of marshal_df. It takes separate
    variable parameter arrays and a constant parameter DataFrame, and combines
    them into a single DataFrame. It handles the correct assignment of
    stratified variable parameters.

    Args:
        var_vals: List of numpy arrays, each containing values for one variable
            parameter.
        const_vals: DataFrame containing all constant parameter values.
        var_params: List of variable parameter names, corresponding to the
            arrays in var_vals.
        stratifiers: Dictionary mapping variable parameter names to their
            stratification columns. If a parameter is not stratified, it should
            not be included in this dictionary.

    Returns:
        A DataFrame combining all variable and constant parameters. The
        resulting DataFrame will have the same number of rows as const_vals,
        with variable parameters correctly assigned based on their
        stratification.
    """
    df = const_vals.copy()
    for var_param, var_val in zip(var_params, var_vals):
        stratifier_mult: str = stratifiers.get(var_param)  # type: ignore
        multiplex(var_val, var_param, stratifier_mult, df)
    return df


def separate_var_vals(
    var_vals: np.ndarray,
    const_vals: pd.DataFrame,
    var_params: list[str],
    stratifiers: Dict[str, str],
) -> list[np.ndarray]:
    var_vals_separate: list[np.ndarray] = []
    start_idx = 0
    for var_param in var_params:
        stratifier = stratifiers.get(var_param)
        if stratifier is None:
            var_vals_separate.append(var_vals[start_idx : start_idx + 1])
            start_idx += 1
        else:
            n_vals = len(const_vals[stratifier].unique())
            var_vals_separate.append(var_vals[start_idx : start_idx + n_vals])
            start_idx += n_vals
    return var_vals_separate


def assemble_bounds_arrs(
    param_bounds: Dict[str, tuple[float, float]],
    var_params: list[str],
    const_vals: pd.DataFrame,
    stratifiers: Dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assembles arrays of lower and upper bounds for optimization.

    Args:
        param_bounds: Dictionary of parameter bounds.
        var_params: List of variable parameter names.
        const_vals: DataFrame of constant parameter values.
        stratifiers: Dictionary mapping parameters to their stratification columns.

    Returns:
        Tuple of lower and upper bound arrays.
    """
    var_bot_bound: list[float] = []
    var_top_bound: list[float] = []
    for var_param in var_params:
        stratifier = stratifiers.get(var_param)
        if stratifier is None:
            bot_bound, top_bound = param_bounds[var_param]
            var_bot_bound.append(bot_bound)
            var_top_bound.append(top_bound)
        else:
            n_vals = len(const_vals[stratifier].unique())
            bot_bound, top_bound = param_bounds[var_param]
            var_bot_bound.extend([bot_bound] * n_vals)
            var_top_bound.extend([top_bound] * n_vals)
    return np.array(var_bot_bound), np.array(var_top_bound)


def assemble_jac_sparsity(
    var_params: list[str], const_vals: pd.DataFrame, stratifiers: Dict[str, str]
) -> np.ndarray:
    """
    Assembles the Jacobian sparsity matrix for optimization.

    Args:
        var_params: List of variable parameter names.
        const_vals: DataFrame of constant parameter values.
        stratifiers: Dictionary mapping parameters to their stratification columns.

    Returns:
        Jacobian sparsity matrix as a boolean array.
    """
    # (m, n) matrix where m is the number of residuals and n is the number of
    # variables
    n_resids = const_vals.shape[0]
    sparsity = np.zeros((n_resids, 0), dtype=bool)
    for var_param in var_params:
        stratifier = stratifiers.get(var_param)
        if stratifier is None:
            sparsity = np.hstack([sparsity, np.ones((n_resids, 1), dtype=bool)])
        else:
            stratifier_vals = const_vals[stratifier].unique()
            for stratifier_val in stratifier_vals:
                sparsity = np.hstack(
                    [
                        sparsity,
                        (const_vals[stratifier] == stratifier_val).values[
                            :, np.newaxis
                        ],
                    ]
                )
    return sparsity
