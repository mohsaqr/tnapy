"""Example datasets for TNA package.

This module provides access to example datasets for demonstrating
and testing TNA functionality. The datasets are from the R TNA package.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def load_group_regulation() -> pd.DataFrame:
    """Load the group regulation dataset in wide format.

    This dataset contains sequences of collaborative regulation behaviors
    from 2000 learning sessions. Each row represents a session, with columns
    T1, T2, ... containing the sequence of regulation behaviors.

    The regulation behaviors include:
    - cohesion: Group cohesion activities
    - consensus: Reaching consensus
    - discuss: Discussion activities
    - monitor: Monitoring activities
    - plan: Planning activities
    - emotion: Emotional regulation
    - adapt: Adaptation activities
    - synthesis: Synthesis activities
    - coregulate: Co-regulation activities

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame where rows are sessions and columns
        are time steps (T1, T2, etc.)

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation()
    >>> model = tna.tna(df)
    >>> tna.centralities(model)
    """
    data_path = Path(__file__).parent / "group_regulation.csv"
    if data_path.exists():
        return pd.read_csv(data_path)

    raise FileNotFoundError(
        f"group_regulation.csv not found at {data_path}. "
        "Please ensure the data file is included in the package."
    )


def load_group_regulation_long() -> pd.DataFrame:
    """Load the group regulation dataset in long format.

    This dataset contains the same collaborative regulation data but in
    long format with one row per event, suitable for use with prepare_data().

    Columns:
    - Actor: Actor/participant identifier
    - Achiever: Achievement level (High/Low)
    - Group: Group identifier
    - Course: Course identifier
    - Time: Timestamp of the event
    - Action: The regulation behavior

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per event

    Examples
    --------
    >>> import tna
    >>> df = tna.load_group_regulation_long()
    >>> prepared = tna.prepare_data(
    ...     df,
    ...     actor="Actor",
    ...     time="Time",
    ...     action="Action"
    ... )
    >>> model = tna.tna(prepared)
    """
    data_path = Path(__file__).parent / "group_regulation_long.csv"
    if data_path.exists():
        return pd.read_csv(data_path)

    raise FileNotFoundError(
        f"group_regulation_long.csv not found at {data_path}. "
        "Please ensure the data file is included in the package."
    )
