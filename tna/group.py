"""Group TNA models — build and manage per-group transition networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .model import TNA, build_model
from .prepare import TNAData

if TYPE_CHECKING:
    pass


def _is_group_tna(x: Any) -> bool:
    """Duck-type check for GroupTNA (avoids circular imports)."""
    return hasattr(x, "models") and hasattr(x, "items") and callable(getattr(x, "items", None))


@dataclass
class GroupTNA:
    """Container for grouped TNA models.

    A dict-like container mapping group names to TNA model objects.
    Supports dict-style access (``group_model["High"]``), iteration,
    ``len()``, ``keys()``, ``values()``, and ``items()``.

    Attributes
    ----------
    models : dict of str to TNA
        Ordered mapping from group name to TNA model.
    """

    models: dict[str, TNA]

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> TNA:
        return self.models[key]

    def __contains__(self, key: str) -> bool:
        return key in self.models

    def __iter__(self):
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)

    def keys(self):
        return self.models.keys()

    def values(self):
        return self.models.values()

    def items(self):
        return self.models.items()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        groups = list(self.models.keys())
        return f"GroupTNA(groups={groups})"

    def __str__(self) -> str:
        lines = [f"GroupTNA with {len(self.models)} groups:"]
        for name, model in self.models.items():
            n_edges = int(np.sum(model.weights > 0))
            lines.append(
                f"  {name}: {len(model.labels)} states, {n_edges} edges"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        """Return group names as a list."""
        return list(self.models.keys())

    def rename_groups(self, new_names: list[str]) -> "GroupTNA":
        """Return a new GroupTNA with renamed groups.

        Parameters
        ----------
        new_names : list of str
            New names, one per existing group (same order).

        Returns
        -------
        GroupTNA
        """
        old_names = list(self.models.keys())
        if len(new_names) != len(old_names):
            raise ValueError(
                f"Expected {len(old_names)} names, got {len(new_names)}"
            )
        new_models = {
            new: self.models[old] for old, new in zip(old_names, new_names)
        }
        return GroupTNA(models=new_models)

    def summary(self) -> pd.DataFrame:
        """Return summary statistics for each group.

        Returns
        -------
        pd.DataFrame
            One row per group with network summary statistics.
        """
        rows = []
        for name, model in self.models.items():
            s = model.summary()
            s["group"] = name
            rows.append(s)
        return pd.DataFrame(rows).set_index("group")

    def apply(self, func, *args, **kwargs) -> dict:
        """Apply *func* to each group's TNA model.

        Parameters
        ----------
        func : callable
            Function whose first argument is a TNA model.
        *args, **kwargs
            Additional arguments forwarded to *func*.

        Returns
        -------
        dict of str to Any
            Mapping from group name to the function result.
        """
        return {name: func(m, *args, **kwargs) for name, m in self.models.items()}


# ======================================================================
# Builder helpers
# ======================================================================


def _build_group_models(
    data: pd.DataFrame | TNAData,
    group: str | list | np.ndarray,
    type_: str = "relative",
    scaling: str | list[str] | None = None,
    labels: list[str] | None = None,
    **kwargs: Any,
) -> GroupTNA:
    """Build one TNA model per group level.

    Parameters
    ----------
    data : pd.DataFrame or TNAData
        Input data (wide-format sequences or TNAData from prepare_data).
    group : str, list, or np.ndarray
        Grouping variable.
        * *str* — column name in ``TNAData.meta_data`` or in a wide-format
          DataFrame.
        * *array-like* — one group label per sequence row.
    type_ : str
        Model type (``"relative"``, ``"frequency"``, etc.).
    scaling : str or list of str, optional
        Scaling to apply to each group model.
    labels : list of str, optional
        Shared state labels across all groups.  If ``None``, labels are
        auto-detected from the full data so every group shares the same
        state space.
    **kwargs
        Additional arguments forwarded to :func:`build_model`.

    Returns
    -------
    GroupTNA
    """
    # --- Resolve sequence data and group vector -----------------------
    if isinstance(data, TNAData):
        seq_data = data.sequence_data
        meta_data = data.meta_data

        if isinstance(group, str):
            if group not in meta_data.columns:
                raise ValueError(
                    f"Column '{group}' not found in metadata. "
                    f"Available: {list(meta_data.columns)}"
                )
            # Align metadata with sequence data using shared index
            group_values = meta_data.loc[seq_data.index, group].values
        else:
            group_arr = np.asarray(group)
            if len(group_arr) != len(seq_data):
                raise ValueError(
                    f"Group vector length ({len(group_arr)}) doesn't match "
                    f"number of sequences ({len(seq_data)})"
                )
            group_values = group_arr

    elif isinstance(data, pd.DataFrame):
        if isinstance(group, str):
            if group not in data.columns:
                raise ValueError(
                    f"Column '{group}' not found in DataFrame. "
                    f"Available: {list(data.columns)}"
                )
            group_values = data[group].values
            seq_data = data.drop(columns=[group])
        else:
            group_arr = np.asarray(group)
            if len(group_arr) != len(data):
                raise ValueError(
                    f"Group vector length ({len(group_arr)}) doesn't match "
                    f"number of rows ({len(data)})"
                )
            group_values = group_arr
            seq_data = data
    else:
        raise TypeError(f"Expected TNAData or DataFrame, got {type(data)}")

    # --- Detect shared labels across all groups -----------------------
    if labels is None:
        if isinstance(seq_data, pd.DataFrame):
            flat = seq_data.values.flatten()
        else:
            flat = np.asarray(seq_data).flatten()
        unique_states = pd.unique(flat[~pd.isna(flat)])
        labels = sorted([str(s) for s in unique_states])

    # --- Build one model per group (preserve appearance order) --------
    unique_groups = list(dict.fromkeys(group_values))

    models: dict[str, TNA] = {}
    for grp in unique_groups:
        mask = group_values == grp
        if isinstance(seq_data, pd.DataFrame):
            grp_data = seq_data.loc[mask]
        else:
            grp_data = seq_data[mask]

        model = build_model(
            grp_data, type_=type_, scaling=scaling, labels=labels, **kwargs
        )
        models[str(grp)] = model

    return GroupTNA(models=models)


# ======================================================================
# Public convenience builders
# ======================================================================


def group_tna(
    x: pd.DataFrame | TNAData,
    group: str | list | np.ndarray,
    scaling: str | list[str] | None = None,
    labels: list[str] | None = None,
    **kwargs: Any,
) -> GroupTNA:
    """Build grouped relative transition probability models.

    Parameters
    ----------
    x : pd.DataFrame or TNAData
        Input data.
    group : str, list, or np.ndarray
        Grouping variable (column name or array).
    scaling : str or list of str, optional
        Scaling to apply.
    labels : list of str, optional
        Shared state labels.
    **kwargs
        Additional arguments forwarded to :func:`build_model`.

    Returns
    -------
    GroupTNA

    Examples
    --------
    >>> import tna
    >>> prepared = tna.prepare_data(long_df, action="Action", actor="Actor", time="Time")
    >>> gm = tna.group_tna(prepared, group="Achiever")
    >>> gm["High"]  # access individual group model
    """
    return _build_group_models(
        x, group, type_="relative", scaling=scaling, labels=labels, **kwargs
    )


def group_ftna(
    x: pd.DataFrame | TNAData,
    group: str | list | np.ndarray,
    scaling: str | list[str] | None = None,
    labels: list[str] | None = None,
    **kwargs: Any,
) -> GroupTNA:
    """Build grouped frequency-based transition models."""
    return _build_group_models(
        x, group, type_="frequency", scaling=scaling, labels=labels, **kwargs
    )


def group_ctna(
    x: pd.DataFrame | TNAData,
    group: str | list | np.ndarray,
    scaling: str | list[str] | None = None,
    labels: list[str] | None = None,
    **kwargs: Any,
) -> GroupTNA:
    """Build grouped co-occurrence transition models."""
    return _build_group_models(
        x, group, type_="co-occurrence", scaling=scaling, labels=labels, **kwargs
    )


def group_atna(
    x: pd.DataFrame | TNAData,
    group: str | list | np.ndarray,
    scaling: str | list[str] | None = None,
    labels: list[str] | None = None,
    **kwargs: Any,
) -> GroupTNA:
    """Build grouped attention-weighted transition models."""
    return _build_group_models(
        x, group, type_="attention", scaling=scaling, labels=labels, **kwargs
    )
