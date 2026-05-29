"""Reference implementation for design-based IPW confidence sequences.

The input is one row per experimental unit with columns:

    entry_time, treatment, propensity, outcome_time, outcome_value

The output has one row per input row, sorted by entry_time.  Each row evaluates
the cumulative reward estimators and confidence sequences at that entry time.
Missing outcome_time or outcome_value values are treated as right-censored, so
they do not contribute to the cumulative reward by any reported time.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "entry_time",
    "treatment",
    "propensity",
    "outcome_time",
    "outcome_value",
]


def normal_mixture_boundary(variance, alpha: float, eta2: float = 1.0):
    """Evaluate the normal-mixture boundary.

    The boundary is

        sqrt(((V * eta2 + 1) / eta2) * log((V * eta2 + 1) / alpha**2)).

    Parameters
    ----------
    variance:
        Nonnegative scalar or array-like variance clock.
    alpha:
        Error probability in (0, 1].
    eta2:
        Positive mixing variance parameter.
    """

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    if eta2 <= 0:
        raise ValueError("eta2 must be positive.")

    variance = np.asarray(variance, dtype=float)
    if np.any(variance < 0):
        raise ValueError("variance must be nonnegative.")

    return np.sqrt(((variance * eta2 + 1.0) / eta2) * np.log((variance * eta2 + 1.0) / alpha**2))


def ipw_confidence_sequences(
    data: pd.DataFrame,
    *,
    alpha: float = 0.05,
    eta2: float = 1.0,
    treatment_value=1,
    control_value=0,
    include_p_value: bool = True,
) -> pd.DataFrame:
    """Compute design-based IPW confidence sequences for delayed outcomes.

    Parameters
    ----------
    data:
        A dataframe with one row per unit and columns:

        - ``entry_time``: time at which the unit entered the experiment.
        - ``treatment``: realized treatment assignment.
        - ``propensity``: probability of treatment, pi_i(1).
        - ``outcome_time``: observed event time. Missing values are treated as
          right-censored.
        - ``outcome_value``: observed reward at ``outcome_time``. Missing values
          are treated as right-censored.

    alpha:
        Error probability for the treatment-effect confidence sequence.
    eta2:
        Positive mixing variance parameter for the normal-mixture boundary.
    treatment_value, control_value:
        Values in the ``treatment`` column identifying treatment and control.
    include_p_value:
        Whether to include the sequential p-value for testing Delta_t = 0.

    Returns
    -------
    pandas.DataFrame
        One row per input row, sorted by ``entry_time``.  Estimates are evaluated
        at the row's entry time.
    """

    _validate_input(data, alpha, eta2, treatment_value, control_value)

    df = data.copy()
    is_datetime = _is_datetime_like(df["entry_time"]) or _is_datetime_like(df["outcome_time"])
    df["entry_time"] = _coerce_time_series(df["entry_time"], is_datetime)
    df["outcome_time"] = _coerce_time_series(df["outcome_time"], is_datetime)
    df["outcome_value"] = pd.to_numeric(df["outcome_value"], errors="coerce")
    df["propensity"] = pd.to_numeric(df["propensity"], errors="coerce")
    df["_original_index"] = np.arange(len(df))

    if df["entry_time"].isna().any():
        raise ValueError("entry_time must be observed for every row.")
    if df["propensity"].isna().any() or (df["propensity"] <= 0).any() or (df["propensity"] >= 1).any():
        raise ValueError("propensity must be strictly between 0 and 1.")

    df = df.sort_values(["entry_time", "_original_index"], kind="mergesort").reset_index(drop=True)
    evaluation_times = df["entry_time"]

    event_observed = _observed_event_mask(df, is_datetime)
    events = df.loc[event_observed].copy()

    if len(events) > 0:
        event_increments = _event_increments(events, treatment_value, control_value)
        event_increments = event_increments.sort_values("outcome_time", kind="mergesort")
    else:
        event_increments = pd.DataFrame(
            columns=[
                "outcome_time",
                "control_events",
                "treatment_events",
                "control_estimate_increment",
                "treatment_estimate_increment",
                "control_variance_increment",
                "treatment_variance_increment",
            ]
        )

    output = _evaluate_at_entry_times(df, evaluation_times, event_increments)
    arm_alpha = alpha / 2.0

    output["control_boundary"] = normal_mixture_boundary(output["control_variance"], arm_alpha, eta2)
    output["treatment_boundary"] = normal_mixture_boundary(output["treatment_variance"], arm_alpha, eta2)
    output["control_lower"] = output["control_estimate"] - output["control_boundary"]
    output["control_upper"] = output["control_estimate"] + output["control_boundary"]
    output["treatment_lower"] = output["treatment_estimate"] - output["treatment_boundary"]
    output["treatment_upper"] = output["treatment_estimate"] + output["treatment_boundary"]

    output["effect_estimate"] = output["treatment_estimate"] - output["control_estimate"]
    output["effect_boundary"] = output["control_boundary"] + output["treatment_boundary"]
    output["effect_lower"] = output["effect_estimate"] - output["effect_boundary"]
    output["effect_upper"] = output["effect_estimate"] + output["effect_boundary"]

    if include_p_value:
        output["p_value"] = [
            _sequential_p_value(abs(effect), v0, v1, eta2)
            for effect, v0, v1 in zip(
                output["effect_estimate"],
                output["control_variance"],
                output["treatment_variance"],
            )
        ]

    return output


def _validate_input(data, alpha, eta2, treatment_value, control_value) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    if eta2 <= 0:
        raise ValueError("eta2 must be positive.")
    if treatment_value == control_value:
        raise ValueError("treatment_value and control_value must be distinct.")
    allowed = {treatment_value, control_value}
    unknown = set(data["treatment"].dropna().unique()) - allowed
    if unknown:
        raise ValueError(f"treatment contains values other than treatment/control: {sorted(unknown)}")


def _is_datetime_like(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _coerce_time_series(series: pd.Series, is_datetime: bool) -> pd.Series:
    if is_datetime:
        return pd.to_datetime(series, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _observed_event_mask(df: pd.DataFrame, is_datetime: bool) -> pd.Series:
    if is_datetime:
        finite_time = df["outcome_time"].notna()
    else:
        finite_time = df["outcome_time"].notna() & np.isfinite(df["outcome_time"])
    finite_value = df["outcome_value"].notna()
    after_entry = df["outcome_time"] >= df["entry_time"]
    return finite_time & finite_value & after_entry


def _event_increments(events: pd.DataFrame, treatment_value, control_value) -> pd.DataFrame:
    treatment = events["treatment"] == treatment_value
    control = events["treatment"] == control_value
    pi1 = events["propensity"].astype(float)
    pi0 = 1.0 - pi1
    y = events["outcome_value"].astype(float)

    increments = pd.DataFrame({"outcome_time": events["outcome_time"]})
    increments["control_events"] = control.astype(int).to_numpy()
    increments["treatment_events"] = treatment.astype(int).to_numpy()
    increments["control_estimate_increment"] = np.where(control, y / pi0, 0.0)
    increments["treatment_estimate_increment"] = np.where(treatment, y / pi1, 0.0)
    increments["control_variance_increment"] = np.where(control, (1.0 - pi0) * (y / pi0) ** 2, 0.0)
    increments["treatment_variance_increment"] = np.where(treatment, (1.0 - pi1) * (y / pi1) ** 2, 0.0)
    return increments


def _evaluate_at_entry_times(
    sorted_units: pd.DataFrame,
    evaluation_times: pd.Series,
    event_increments: pd.DataFrame,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "time": evaluation_times,
            "entry_time": sorted_units["entry_time"],
            "n_entered": _n_entered_at_times(sorted_units["entry_time"], evaluation_times),
        }
    )

    cumulative_columns = {
        "control_events": "control_events",
        "treatment_events": "treatment_events",
        "control_estimate_increment": "control_estimate",
        "treatment_estimate_increment": "treatment_estimate",
        "control_variance_increment": "control_variance",
        "treatment_variance_increment": "treatment_variance",
    }

    for source, target in cumulative_columns.items():
        output[target] = _cumulative_event_sum(
            event_increments["outcome_time"],
            event_increments[source],
            evaluation_times,
        )

    return output.reset_index(drop=True)


def _n_entered_at_times(entry_times: pd.Series, evaluation_times: pd.Series) -> np.ndarray:
    values = entry_times.to_numpy()
    return np.searchsorted(values, evaluation_times.to_numpy(), side="right")


def _cumulative_event_sum(event_times: pd.Series, increments: pd.Series, evaluation_times: pd.Series) -> np.ndarray:
    if len(event_times) == 0:
        return np.zeros(len(evaluation_times))

    order = np.argsort(event_times.to_numpy(), kind="mergesort")
    sorted_event_times = event_times.to_numpy()[order]
    sorted_increments = increments.astype(float).to_numpy()[order]
    cumulative = np.cumsum(sorted_increments)
    indices = np.searchsorted(sorted_event_times, evaluation_times.to_numpy(), side="right") - 1
    result = np.zeros(len(evaluation_times))
    observed = indices >= 0
    result[observed] = cumulative[indices[observed]]
    return result


def _sequential_p_value(abs_effect: float, control_variance: float, treatment_variance: float, eta2: float) -> float:
    """Solve |effect_estimate| = b(V0; p/2) + b(V1; p/2)."""

    if not math.isfinite(abs_effect):
        return np.nan

    largest_alpha_boundary = normal_mixture_boundary(control_variance, 0.5, eta2) + normal_mixture_boundary(
        treatment_variance, 0.5, eta2
    )
    if abs_effect <= largest_alpha_boundary:
        return 1.0

    lo = np.finfo(float).tiny
    hi = 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        boundary = normal_mixture_boundary(control_variance, mid / 2.0, eta2) + normal_mixture_boundary(
            treatment_variance, mid / 2.0, eta2
        )
        if boundary > abs_effect:
            lo = mid
        else:
            hi = mid
    return hi
